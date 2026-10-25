import React, { useState } from "react";
import { Archive, ArrowRight, Check, Pencil, Play, RotateCcw, Save, X } from "lucide-react";
import { GalleryBatch } from "./api";

type Props = {
  batches: GalleryBatch[];
  busy: boolean;
  open: (id: string) => void;
  resume: (id: string) => void;
  manage: (id: string, action: string, name?: string) => Promise<void>;
};

export function GalleryBatches({ batches, busy, open, resume, manage }: Props) {
  const [filter, setFilter] = useState("active");
  const [editing, setEditing] = useState("");
  const [name, setName] = useState("");
  const visible = batches.filter((b) => filter === "all" || (filter === "archived" ? b.archived : !b.archived));
  return <div className="batch-manager">
    <label className="field"><span>Batch status</span>
      <select aria-label="Batch status" value={filter} onChange={(e) => setFilter(e.target.value)}>
        <option value="active">Not archived</option><option value="archived">Archived</option><option value="all">All batches</option>
      </select>
    </label>
    {!visible.length && <p className="muted">No batches in this view.</p>}
    {visible.map((batch) => {
      const finished = ["complete", "exhausted"].includes(batch.status);
      return <section className="batch-row" key={batch.id} aria-label={"Batch " + batch.name}>
        <div className="batch-heading">
          {editing === batch.id ? <form onSubmit={async (e) => { e.preventDefault(); await manage(batch.id, "rename", name); setEditing(""); }}>
            <input aria-label="Batch name" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} />
            <button className="icon-button" title="Save batch name" disabled={busy || !name.trim()}><Save size={16} /></button>
            <button type="button" className="icon-button" title="Cancel rename" onClick={() => setEditing("")}><X size={16} /></button>
          </form> : <><h3>{batch.name}</h3><button className="icon-button" title="Rename batch" disabled={busy} onClick={() => {setName(batch.name); setEditing(batch.id);}}><Pencil size={15} /></button></>}
          <span className="status">{batch.archived ? "Archived" : batch.status.replaceAll("_", " ")}</span>
        </div>
        <p className="muted batch-meta">{batch.created_at} · {batch.id}</p>
        {batch.theme && <p>{batch.theme}</p>}
        <div className="batch-counts">
          {[["Found", batch.counts.found], ["Approved", batch.counts.approved], ["Rejected", batch.counts.rejected], ["Generated", `${batch.counts.generated}/${batch.quantity}`], ["Need review", batch.counts.review], ["Pending", batch.counts.pending]].map(([label, value]) => <span key={label}><strong>{value}</strong> {label}</span>)}
        </div>
        {batch.message && <p role="status">{batch.message}</p>}
        {batch.last_error && <p className="error-text">{batch.last_error}</p>}
        {batch.completion_reason && <p className="muted">{batch.completion_reason.replaceAll("_", " ")}</p>}
        {batch.legacy_shared_references && <p className="muted">Legacy findings; ownership will be isolated when resumed.</p>}
        <div className="button-row">
          <button onClick={() => open(batch.id)}><ArrowRight size={16} /> Open batch</button>
          {!finished && !batch.archived && <button disabled={busy} onClick={() => resume(batch.id)}><Play size={16} /> Resume batch</button>}
          {!finished && !batch.archived && <button disabled={busy} onClick={() => {if (confirm("Finish this batch without generating remaining items? Existing images and references will be retained.")) void manage(batch.id, "finish");}}><Check size={16} /> Finish batch</button>}
          <button disabled={busy} onClick={() => {if (batch.archived || confirm("Archive this batch? Nothing will be deleted; restore it to resume later.")) void manage(batch.id, batch.archived ? "restore" : "archive");}}>
            {batch.archived ? <RotateCcw size={16} /> : <Archive size={16} />}{batch.archived ? "Restore batch" : "Archive batch"}
          </button>
        </div>
      </section>;
    })}
  </div>;
}
