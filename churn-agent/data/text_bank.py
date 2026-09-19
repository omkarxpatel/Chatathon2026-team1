"""Free-text bank for support tickets.

Split out of generator.py on purpose: this is the unstructured signal the
LLM layer actually reads, so it is the file you want to enrich when the
agent's diagnoses look thin. Editing it requires no knowledge of the
simulation logic.

Keyed by (TicketCategory, tone) where tone is "neg" | "neu" | "pos".
"""

from __future__ import annotations

from data.schema import TicketCategory as TC

TICKET_TEXT: dict[tuple[str, str], list[str]] = {
    (TC.DAMAGED_SHIPMENT, "neg"): [
        "Second tub in a row arrived with the seal broken and powder all through the box. "
        "I've now got protein dust over everything. This is getting old.",
        "Box was crushed and the lid had popped off. Half the creatine is gone. "
        "I paid full price for this and I'd like it sorted out properly.",
        "My order turned up with the shaker cracked and the whey leaking. "
        "I sent photos last week and nobody has got back to me.",
    ],
    (TC.DAMAGED_SHIPMENT, "neu"): [
        "The outer box was dented but the tub inside seems fine. "
        "Flagging it in case it's a packaging issue on your end.",
    ],
    (TC.SHIPPING_DELAY, "neg"): [
        "Ordered 12 days ago, tracking hasn't moved since it left your warehouse. "
        "I've been out of protein for a week.",
        "This is the third order that's been late. I train every morning and "
        "I can't keep guessing when it'll show up.",
    ],
    (TC.SHIPPING_DELAY, "neu"): [
        "Any update on my order? Tracking says in transit but no movement for 4 days.",
        "Is standard shipping usually this slow to the west coast?",
    ],
    (TC.PRODUCT_QUESTION, "neu"): [
        "Quick question - should I be taking creatine on rest days too, or only when I lift?",
        "Does the pre-workout have caffeine? Trying to avoid it in the evenings.",
        "How many scoops of the electrolytes for a long ride in hot weather?",
        "I've torn my rotator cuff and I'm out of the gym for at least three months. "
        "Is there any point continuing the creatine while I'm not training?",
        "Moving overseas for work in a few weeks - do you ship internationally?",
    ],
    (TC.PRODUCT_QUESTION, "pos"): [
        "Love the vanilla, any chance of a cinnamon flavour? Would buy a 5 lb tub immediately.",
    ],
    (TC.TASTE_MIXABILITY, "neg"): [
        "The new chocolate batch is chalky and won't dissolve properly. "
        "The old formula mixed fine. Did something change?",
        "Honestly the taste has gone downhill. I finished the tub but I'm not reordering this flavour.",
    ],
    (TC.TASTE_MIXABILITY, "neu"): [
        "Any tips for mixing the creatine? It settles at the bottom of my bottle.",
    ],
    (TC.BILLING, "neg"): [
        "I was charged twice for the same order. Please refund one of them.",
        "The promo code from your email didn't apply at checkout and I got charged full price. "
        "That's the only reason I was buying now.",
    ],
    (TC.BILLING, "neu"): [
        "Do you offer a subscribe-and-save price? Paying full retail every month adds up "
        "and I'm comparing against a couple of other brands right now.",
        "Can I get a VAT receipt for my last three orders?",
    ],
    (TC.RETURN_REFUND, "neg"): [
        "I'd like to return the unopened pre-workout. It's too strong for me and "
        "I'm not going to use it.",
        "Requesting a refund on my last order. Product arrived past its best-before date.",
    ],
    (TC.PRAISE, "pos"): [
        "Just wanted to say the isolate is the best I've used - no bloating at all. "
        "Been on it six months now.",
        "Your support team sorted my delivery issue in under an hour. Genuinely impressed.",
        "Down 14 lbs and the electrolytes have been a big part of getting through "
        "the training. Thanks for making a product that actually works.",
    ],
}

# Fallback so the generator never crashes on an unseeded (category, tone).
_FALLBACK = {
    "neg": ["Not happy with my recent order and would like someone to look into it."],
    "neu": ["Had a question about my recent order."],
    "pos": ["Happy with the product, just wanted to pass that on."],
}


def pick(category, tone: str, rng) -> str:
    options = TICKET_TEXT.get((category, tone))
    if not options:
        # Fall back across tones within the category before giving up.
        for alt in ("neu", "neg", "pos"):
            options = TICKET_TEXT.get((category, alt))
            if options:
                break
    if not options:
        options = _FALLBACK[tone]
    return str(rng.choice(options))


# Broadcast sends go to the whole list. They are NOT subject to the
# retention cooldown -- a newsletter is not outreach.
CAMPAIGN_TYPES = [
    "weekly_newsletter",
    "new_flavour_launch",
    "training_tips",
    "seasonal_promo",
    "loyalty_update",
]

# Targeted, one-to-one sends triggered by a flow. THESE are what the
# cooldown gate counts: emailing someone a win-back two days after the last
# win-back is exactly the pushy behaviour we are trying to prevent.
TARGETED_CAMPAIGN_TYPES = [
    "winback_flow",
    "replenishment_reminder",
    "vip_checkin",
    "service_followup",
]
