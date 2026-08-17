"""设备指纹生成 —— 用于 Cloudflare Worker 分发的设备标识（V1.0）。

把本机多个稳定特征拼起来做 SHA-256，得到一个 64 位十六进制指纹。
任何单个特征获取失败都用占位串降级，保证：

- 指纹总能生成，不会让客户端卡死；
- 同一台机器多次运行结果稳定（否则每日 20 次限额会错乱）；
- 不同机器大概率不同（用于「设备绑定」与「每设备每日限额」）。

注意：指纹只用于额度统计与 Key 归属，不用于身份认证；被克隆硬盘的
两台机器会共享同一个指纹（文档已声明这是设计如此）。
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys

try:
    import getpass

    def _username() -> str:
        try:
            return getpass.getuser()
        except Exception:  # noqa: BLE001 —— 无登录会话时降级
            return "unknown_user"

except Exception:  # noqa: BLE001
    def _username() -> str:
        return "unknown_user"


# 版本盐：一旦要全局更换指纹规则，改这个值即可（旧 Key 自动作废）。
_FINGERPRINT_VERSION = "recallos-v1"

_VIRTUAL_IFACE_KEYWORDS = (
    "lo",
    "docker",
    "veth",
    "br-",
    "virbr",
    "vmnet",
    "vbox",
    "tun",
    "tap",
)

_MAC_PATTERN = re.compile(
    r"([0-9a-fA-F]{2}[:.-][0-9a-fA-F]{2}[:.-][0-9a-fA-F]{2}"
    r"[:.-][0-9a-fA-F]{2}[:.-][0-9a-fA-F]{2}[:.-][0-9a-fA-F]{2})"
)


def get_device_fingerprint() -> str:
    """生成 64 位十六进制设备指纹。"""
    components = [
        _get_disk_serial(),
        _get_mac_address(),
        _username(),
        platform.node() or "unknown_host",
        sys.platform,
        _FINGERPRINT_VERSION,
    ]
    return hashlib.sha256("|".join(components).encode("utf-8")).hexdigest()


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """执行外部命令并返回 stdout；任何失败返回空串。"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout or ""
    except Exception:  # noqa: BLE001 —— 指纹获取永不抛错
        return ""


def _get_disk_serial() -> str:
    """获取主硬盘序列号；拿不到就返回占位串。"""
    try:
        if sys.platform == "win32":
            for line in _run(["wmic", "diskdrive", "get", "serialnumber"]).splitlines():
                serial = line.strip()
                if serial and serial.lower() not in ("serialnumber", ""):
                    return serial
            # wmic 在新系统已被弃用/移除：用卷标卷序列号兜底
            for line in _run(["cmd", "/c", "vol", "C:"]).splitlines():
                match = re.search(r"([A-F0-9]{4}-[A-F0-9]{4})", line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
        elif sys.platform == "darwin":
            for line in _run(["system_profiler", "SPHardwareDataType"]).splitlines():
                if "Hardware UUID" in line:
                    return line.split(":")[-1].strip()
        else:
            # Linux：machine-id 是比 /etc/machine-id 更稳的系统标识
            for path in ("/var/lib/dbus/machine-id", "/etc/machine-id"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            return content
                except OSError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    return "unknown_disk"


def _get_mac_address() -> str:
    """获取第一个非虚拟网卡的 MAC 地址；拿不到就返回占位串。"""
    try:
        mac = _mac_from_psutil()
        if mac:
            return mac
        if sys.platform == "win32":
            for line in _run(["ipconfig", "/all"]).splitlines():
                if "Physical Address" in line:
                    mac = line.split(":", 1)[-1].strip()
                    normalized = mac.replace("-", ":").replace(".", ":")
                    if normalized and normalized != "00:00:00:00:00:00":
                        return normalized.upper()
        else:
            cmd = ["ifconfig"] if sys.platform == "darwin" else ["ip", "link"]
            for line in _run(cmd).splitlines():
                match = _MAC_PATTERN.search(line)
                if match:
                    return match.group(1).replace("-", ":").replace(".", ":").upper()
    except Exception:  # noqa: BLE001
        pass
    return "unknown_mac"


def _mac_from_psutil() -> str:
    """可选用 psutil 取 MAC（不引入硬依赖）；失败返回空串。"""
    try:
        import psutil  # type: ignore[import-not-found]

        for iface, addrs in psutil.net_if_addrs().items():
            if _is_virtual_interface(iface):
                continue
            for addr in addrs:
                if getattr(addr, "address", None) and ":" in addr.address:
                    return addr.address.upper()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _is_virtual_interface(name: str) -> bool:
    """判断网卡名是否为虚拟网卡。"""
    lowered = (name or "").lower()
    return any(kw in lowered for kw in _VIRTUAL_IFACE_KEYWORDS)
