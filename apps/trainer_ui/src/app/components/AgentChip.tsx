// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { AgentActivityState } from "../hooks/useAgentActivity";

/** "An agent is labelling" -- shown in the header while the MCP bridge writes.

    Colour is never the only signal, and blue is never paired with purple:
    the chip has a pulsing dot and the words, and uses the vermilion the
    rest of the UI already reserves for non-blue emphasis. */
export default React.memo(function AgentChip({ agent, follow, onToggleFollow }: {
  agent: AgentActivityState;
  /** jump the annotator to whatever the bridge just wrote */
  follow?: boolean;
  onToggleFollow?: () => void;
}) {
  const { t } = useI18n();
  if (!agent.active || !agent.last) return null;
  const e = agent.last;
  const what = e.item_id ?? (e.count != null ? t("agent.items").replace("{n}", String(e.count)) : "");
  const title = t("agent.labelling.desc").replace("{agent}", e.agent).replace("{tool}", e.tool)
    + (what ? ` — ${what}` : "");
  // Compact on purpose: the header centres the project name and a long chip
  // walks into it. The item lands in the tooltip; the list shows it anyway.
  return (
    <div className="agent-chip" role="status" aria-live="polite" title={title} data-desc={title} data-desc-pos="bottom">
      <span className="agent-chip-dot" aria-hidden="true" />
      <span className="agent-chip-label">{t("agent.labelling")}</span>
      <span className="agent-chip-detail">{e.tool}</span>
      {onToggleFollow && (
        <button
          type="button"
          className={"agent-live" + (follow ? " on" : " off")}
          aria-pressed={!!follow}
          aria-label={t(follow ? "agent.live.on" : "agent.live.off")}
          onClick={onToggleFollow}
          title={t("agent.live.desc")}
          data-desc={t("agent.live.desc")}
          data-desc-pos="bottom"
        >
          {/* the state is in the word and the mark, not the colour alone */}
          <span className="agent-live-mark" aria-hidden="true">{follow ? "●" : "○"}</span>
          <span className="agent-live-text">LIVE</span>
          <span className="agent-live-state">{follow ? "ON" : "OFF"}</span>
        </button>
      )}
    </div>
  );
});
