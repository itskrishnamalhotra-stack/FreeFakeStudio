import React, {useEffect, useRef} from "react";
import { Plus, X, Search, Images, RotateCcw, ListChecks } from "lucide-react";
import type { ResourceTab, Job } from "./api";

type Props = {
  tabs: ResourceTab[];
  selected: string;
  busy: boolean;
  jobs: Job[];
  select: (id: string) => void;
  search: (tab?: ResourceTab) => void;
  generate: (tab: ResourceTab) => void;
  review: (tab: ResourceTab) => void;
  remove: (tab: ResourceTab) => void;
  resume: (job: Job) => void;
};

export function ResourceTabs({tabs, selected, busy, jobs, select, search, generate, review, remove, resume}: Props) {
  const strip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = strip.current;
    const active = container?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]');
    if (!container || !active) return;
    const reveal = () => {
      const outer = container.getBoundingClientRect(), inner = active.getBoundingClientRect();
      if (inner.left < outer.left) container.scrollLeft -= outer.left - inner.left;
      else if (inner.right > outer.right) container.scrollLeft += inner.right - outer.right;
    };
    reveal();
    const observer = new ResizeObserver(reveal);
    observer.observe(container);
    return () => observer.disconnect();
  }, [selected, tabs.length]);
  const tab = tabs.find(item => item.id === selected);
  const latest = jobs.find(job => job.payload.resource_tab_id === selected && ["resource_search", "resource_generate", "resource_review"].includes(job.kind));
  const interrupted = latest && ["failed", "interrupted", "cancelled"].includes(latest.status) ? latest : undefined;
  return <div className="resource-workspace">
    <div className="resource-tab-strip">
      <div role="tablist" aria-label="Resource searches" className="resource-tabs" ref={strip}>
        <button role="tab" aria-selected={!selected} onClick={() => select("")}>All resources</button>
        {tabs.map((item, index) => <div key={item.id} className={"resource-tab" + (selected === item.id ? " selected" : "")}>
          <button id={"resource-tab-" + item.id} role="tab" aria-selected={selected === item.id}
            title={item.description || item.name} onClick={() => select(item.id)}
            onKeyDown={event => {
              if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
              event.preventDefault();
              const next = tabs[(index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
              select(next.id); document.getElementById("resource-tab-" + next.id)?.focus();
            }}><span>{item.name}</span><small>{item.counts.found}</small></button>
          <button className="icon-button" aria-label={"Delete resource tab " + item.name} title="Delete resource tab"
            disabled={busy} onClick={() => remove(item)}><X size={14} /></button>
        </div>)}
      </div>
      <button className="icon-button" title="New resource search" aria-label="New resource search" disabled={busy} onClick={() => search()}><Plus size={18} /></button>
    </div>
    {tab ? <div className="resource-tab-details">
      <div className="resource-search-summary"><strong>{tab.name}</strong><p title={tab.description || tab.name}>{tab.description || tab.name}</p>
        <div className="resource-counts"><span><b>{tab.counts.available}</b> Ready unused</span><span><b>{tab.counts.pending}</b> Awaiting checks</span>
          <span><b>{tab.counts.used}</b> Used</span><span><b>{tab.counts.reserved}</b> Reserved</span></div>
        {Boolean(tab.queries?.length) && <details><summary>Search queries</summary><ul>{tab.queries!.map((query, index) => <li key={index}>{query}</li>)}</ul></details>}
      </div>
      <div className="button-row">
        <button disabled={busy || tab.archived || !tab.counts.pending} onClick={() => review(tab)}><ListChecks size={16} /> Review resources</button>
        <button disabled={busy || tab.archived} onClick={() => search(tab)}><Search size={16} /> Search more</button>
        <button className="primary" disabled={busy || tab.archived || !((tab.counts.available || 0) + (tab.counts.pending || 0))}
          onClick={() => generate(tab)}><Images size={16} /> Generate more</button>
        {interrupted && <button disabled={busy} onClick={() => resume(interrupted)}><RotateCcw size={16} /> Resume saved work</button>}
      </div>
      {tab.error && <p className="error-text">{tab.error}</p>}
    </div> : null}
  </div>;
}
