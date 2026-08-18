"""V0.3.1 — 内置精选概念：让用户第一次打开产品就能立即开始学习，无需任何输入。

每个概念自带一段自足的简短原文（约 200 字），可以直接进入
「阅读 → 验证 → 完成」新流程，也方便验证任务（如「用一句话向朋友解释」
这类闭卷任务）只凭原文就能作答。
"""

from __future__ import annotations

BUILTIN_CONCEPTS: list[dict] = [
    {
        "id": "opportunity_cost",
        "title": "Opportunity Cost",
        "category": "Economics",
        "difficulty": 1,
        "hook": "Every choice has a price — and it's the option you gave up.",
        "source_text": (
            "**Opportunity cost** is the value of the best alternative you give up "
            "when you make a choice.\n\n"
            "**In plain words**: nothing is free — every decision hides a future you "
            "decided not to take.\n\n"
            "**For example**: watching TV means you can't read a book at the same time; "
            "saving your money means delaying the joy that money could buy today.\n\n"
            "**The key point**: smart decisions aren't only about what you gain — "
            "they're also about what you give up. Compare your best alternative with "
            "the choice in front of you, and only then judge whether it's worth it."
        ),
    },
    {
        "id": "compound_interest",
        "title": "Compound Interest",
        "category": "Personal Finance",
        "difficulty": 1,
        "hook": "Your returns earn returns — and the snowball grows on its own.",
        "source_text": (
            "**Compound interest** is when the interest you earn is reinvested so it "
            "also earns interest, letting your money grow like a rolling snowball.\n\n"
            "**In plain words**: earnings produce more earnings; compound interest is "
            "a friend of time.\n\n"
            "**For example**: the interest on a savings account, the gains on an index "
            "fund, or the repeated practice of a skill can all compound.\n\n"
            "**The key point**: the three variables that matter most are the principal, "
            "the rate of return, and time. A small principal is fine, a steady return "
            "is precious, and the earlier you start the better — because compounding "
            "needs time to do its work."
        ),
    },
    {
        "id": "survivorship_bias",
        "title": "Survivorship Bias",
        "category": "Mental Models",
        "difficulty": 2,
        "hook": "The failures you never see are the most important evidence.",
        "source_text": (
            "**Survivorship bias** is the error of drawing conclusions from the "
            "survivors only, while ignoring the ones that didn't make it — so the "
            "conclusion ends up distorted.\n\n"
            "**For example**: everyone hears about startups that succeed, but the "
            "thousands that quietly fail are invisible. During WWII, engineers studied "
            "the bullet holes on returning planes and almost added armor where the "
            "holes were — the right place was where there were NO holes, because planes "
            "hit there never made it back to tell their story.\n\n"
            "**The key point**: before you make a judgment, ask yourself — am I seeing "
            "the whole sample, or only the survivors? Fill in the invisible part, and "
            "your conclusion becomes reliable."
        ),
    },
    {
        "id": "marginal_utility",
        "title": "Marginal Utility",
        "category": "Economics",
        "difficulty": 2,
        "hook": "The first bowl of noodles is the most delicious — every extra unit satisfies less.",
        "source_text": (
            "**Marginal utility** is the extra satisfaction you get from consuming one "
            "more unit of something, and it usually decreases as the quantity grows.\n\n"
            "**In plain words**: the first bowl of noodles tastes great; by the third "
            "you're stuffed.\n\n"
            "**For example**: the first shirt you buy excites you; the tenth doesn't — "
            "not because the shirt is bad, but because each extra one has less marginal "
            "value.\n\n"
            "**The key point**: when allocating limited resources, put your time and "
            "money where the marginal utility is highest — the same hour is worth more "
            "when spent on what you lack most. Always look at the difference an "
            "\"extra unit\" makes, not just the total."
        ),
    },
    {
        "id": "sunk_cost",
        "title": "Sunk Cost",
        "category": "Mental Models",
        "difficulty": 1,
        "hook": "Money already spent and unrecoverable shouldn't keep calling the shots.",
        "source_text": (
            "**Sunk cost** is time, money, or effort already spent that can never be "
            "recovered.\n\n"
            "**For example**: tuition already paid, a relationship you've invested "
            "years in, or a project you've poured resources into.\n\n"
            "**The key point**: it triggers the mistake of \"can't let it go\" — you "
            "sit through a terrible movie just because you paid for the ticket. The "
            "rational move: what's past is past; make decisions based on the costs and "
            "benefits of the present and future, and stop letting unrecoverable costs "
            "hold you hostage. Cutting your losses is itself a gain."
        ),
    },
]


def get_builtin_concepts() -> list[dict]:
    """Return copies of all builtin concepts (safe to mutate)."""
    return [dict(c) for c in BUILTIN_CONCEPTS]


def get_builtin_concept(concept_id: str) -> dict | None:
    """Return one builtin concept by id, or None."""
    for c in BUILTIN_CONCEPTS:
        if c["id"] == concept_id:
            return dict(c)
    return None


__all__ = [
    "BUILTIN_CONCEPTS",
    "get_builtin_concept",
    "get_builtin_concepts",
]
