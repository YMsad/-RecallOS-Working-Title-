// ================================================================
// Cloudflare Worker — RecallOS Key 分发服务（V1.0）
// 部署：wrangler deploy （wrangler.toml 见仓库根目录）
// 免费额度：10 万次请求/天，KV 读写免费
// ================================================================
//
// 接口：
//   GET /get-key?device={fingerprint}   → 返回临时 apiKey（绑定设备，24h 有效）
//   GET /revoke?secret={admin}          → 一键撤销：拒绝一切新 Key 请求
//   GET /unrevoke?secret={admin}        → 解除撤销（管理员自查修复后重新开放）
//   GET /health                         → 存活检查
//
// 说明：撤銷通过 KV 里的 global:revoked 标记实现——标记置 1 后，
// 所有 /get-key 请求一律 403（而不是文档初稿里只 bump 版本号却仍发 Key）。
// ================================================================

// ---------- 配置（部署前修改） ----------
const CONFIG = {
  // 有效的 DeepSeek API Key 列表（至少 2 个，按设备指纹哈希轮换分配）
  VALID_KEYS: [
    'sk-your-first-deepseek-api-key',
    'sk-your-second-deepseek-api-key',
  ],

  // 每设备每日 Key 请求上限（客户端缓存 24h，正常每天只会取 1 次）
  DAILY_LIMIT: 20,

  // Key 有效期（秒），默认 24 小时
  KEY_TTL_SECONDS: 86400,

  // 管理员密钥（用于撤销/恢复，务必改成足够随机的长字符串）
  ADMIN_SECRET: 'replace-with-a-long-random-admin-secret',

  // 是否打印调试日志（建议只在联调时打开）
  DEBUG: false,
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (path === '/get-key') return handleGetKey(request, env);
    if (path === '/revoke') return handleRevoke(request, env, true);
    if (path === '/unrevoke') return handleRevoke(request, env, false);
    if (path === '/health') {
      return jsonResponse({ status: 'ok', timestamp: Date.now() });
    }
    return new Response('Not Found', { status: 404 });
  },
};

// ---------- 获取 Key ----------
async function handleGetKey(request, env) {
  const url = new URL(request.url);
  const deviceId = (url.searchParams.get('device') || '').trim();

  // 1. 验证设备指纹：必须为 64 位十六进制
  if (!/^[0-9a-fA-F]{64}$/.test(deviceId)) {
    return jsonResponse({ error: '无效的设备标识', code: 'INVALID_DEVICE' }, 400);
  }

  // 2. 是否已被管理员全局撤销
  if ((await env.KV.get('global:revoked')) === '1') {
    return jsonResponse(
      { error: '服务已暂停，请稍后再试', code: 'REVOKED' },
      403
    );
  }

  // 3. 检查每日限额（KV 存储，24h 后自动过期）
  const today = new Date().toISOString().slice(0, 10);
  const usageKey = `usage:${deviceId}:${today}`;
  const usageCount = parseInt((await env.KV.get(usageKey)) || '0', 10) || 0;

  if (usageCount >= CONFIG.DAILY_LIMIT) {
    return jsonResponse(
      {
        error: '今日学习次数已用尽，明天再来吧',
        code: 'DAILY_LIMIT_EXCEEDED',
        limit: CONFIG.DAILY_LIMIT,
        dailyUsed: usageCount,
      },
      429
    );
  }

  // 4. 计数（先写后读，竞态下最多略超一两次，可接受）
  await env.KV.put(usageKey, String(usageCount + 1), {
    expirationTtl: 86400,
  });

  // 5. 按设备指纹哈希分配 Key：同一设备尽量用同一个 Key，便于在 DeepSeek 侧定位
  const keyIndex = Math.abs(hashString(deviceId)) % CONFIG.VALID_KEYS.length;
  const selectedKey = CONFIG.VALID_KEYS[keyIndex];

  if (CONFIG.DEBUG) {
    console.log(
      `[${today}] 设备 ${deviceId.slice(0, 8)} 取 Key，今日第 ${usageCount + 1} 次`
    );
  }

  // 6. 返回
  return jsonResponse({
    apiKey: selectedKey,
    expiresIn: CONFIG.KEY_TTL_SECONDS,
    expiresAt: new Date(Date.now() + CONFIG.KEY_TTL_SECONDS * 1000).toISOString(),
    dailyLimit: CONFIG.DAILY_LIMIT,
    dailyUsed: usageCount + 1,
  });
}

// ---------- 撤销 / 恢复 ----------
async function handleRevoke(request, env, revoke) {
  const url = new URL(request.url);
  const secret = url.searchParams.get('secret') || '';

  if (secret !== CONFIG.ADMIN_SECRET) {
    return jsonResponse({ error: '未授权', code: 'UNAUTHORIZED' }, 401);
  }

  // 全局版本号：每次撤销/恢复 +1，客户端可用它判断是否强制重取
  const currentVersion = parseInt((await env.KV.get('global:version')) || '0', 10) || 0;
  const newVersion = currentVersion + 1;

  await env.KV.put('global:version', String(newVersion));
  await env.KV.put('global:revoked', revoke ? '1' : '0');

  await env.KV.put(
    'global:revoke_log',
    JSON.stringify({
      timestamp: new Date().toISOString(),
      action: revoke ? 'REVOKE_ALL' : 'UNREVOKE',
      version: newVersion,
    })
  );

  return jsonResponse({
    success: true,
    revoked: revoke,
    version: newVersion,
    message: revoke
      ? '所有 Key 已撤销，新的 /get-key 请求将被拒绝'
      : '服务已恢复',
    timestamp: new Date().toISOString(),
  });
}

// ---------- 辅助 ----------
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash | 0; // 保持 32 位有符号整数
  }
  return hash;
}
