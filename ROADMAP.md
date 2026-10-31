# Roadmap

## Next: Ask-first UI

Ask can now draft agents and dispatches (`propose_agent`, confirmed in a card) and run fresh research (`research_run`), so chat should be the main way in. The manual forms stay, next to the lists they belong to.

Today's entry points (`ui.html`):
- Runs sidebar footer: primary **New dispatch** button.
- Desktop topbar and mobile header: a **New ▾** menu with New dispatch and New agent.
- Tabs: Digest · Dashboard · Ask · Config ▾ (Dispatches, Agents, Settings). Mobile tab bar: Digest · Dashboard · Ask · Runs · Config.
- Config → Agents: per-agent **Schedule** and **Import YAML**. Config → Dispatches: no create button.

Planned:
- [ ] **New chat is the primary action.** The sidebar button and the topbar/header **New ▾** menus become a single **New chat** (`startChat()`: open Ask, start an empty thread, focus the input).
- [ ] **Manual creation next to its list.** Config → Dispatches gets **+ New dispatch**; Config → Agents gets **+ New agent** beside Import YAML; mobile Config gets **+ New** for the current list. Each list has a hint, "Or describe it in Ask", that opens a chat with a draft prompt.
- [ ] **Open on Ask, and put it first.** Tab order Ask · Digest · Dashboard · Config (mobile: Ask · Digest · Dashboard · Runs · Config). First load opens Ask; after that, the last tab is restored.
- [ ] **Today in a new chat.** The empty chat lists today's digests: color dot, label, the one-line summary, and **Ask** / **Open** buttons. About 12 rows, with "show all". Example prompts include "Research the latest on …" and "Track … every weekday morning".
- [ ] Keep: the run drawer's "schedule this" (opens the dispatch form prefilled from a run).

## After that

- [ ] **Measure the dedupe.** How often the cross-digest list is used and followed (`recall` events with `sub=elsewhere`), and whether repeated stories drop. Tune `digests.covered_today` (window, overlap threshold) from real runs.
- [ ] **Recall for feed agents.** Previous-run recall matches URLs, which never repeat for Direct Fetch inbox agents, so `repeat_urls` is always 0 there. Match on key terms instead, the way `covered_today` does.
- [ ] **Summary everywhere.** Show the digest's `Summary:` line in the run drawer and the runs sidebar, not only on digest cards.
- [ ] **Chat-created agents:** edit a drafted agent's card after creation (open the agent form), and list chat-created agents with a `chat` filter in Config.
