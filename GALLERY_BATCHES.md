# Avatar Gallery Batches

Each avatar owns independent saved batches. Open **Gallery > Batches** to see
names, status, target quantity, references found/approved/rejected, generated
images, items needing review and pending work. Use the **Gallery batch** selector
to view only one batch's generations, enhancements and search resources. The
selection is retained in the URL (`?batch=<id>`) and survives a page refresh.
The all-batches view deliberately includes console images too.

## Create and Manage

1. Confirm the avatar's face and body.
2. Select **Create batch**, enter a name, theme and quantity (1-20), and start.
3. **Pipeline** shows findings, rejection reasons, saved prompts and generation
   progress for the selected batch, not combined counts from other batches.
4. **Resume batch** continues that batch's stored settings and pending work.
   Already-saved images are not generated again just because another batch ran.
5. **Finish batch** closes unwanted remaining work without generating it.
   Saved results, prompts and reference files are retained. This is a final close;
   use Archive instead if you may want to resume later.
6. **Archive batch** hides it from the normal management list without deleting
   anything. Select Archived, then **Restore batch**, to make it resumable again.
7. Rename a batch without changing its generation settings or file identifiers.

Only one GPU operation runs at a time. Stop the running operation and wait for its
checkpoint before archiving, finishing or starting another one. An idle paused or
failed batch does not block creation of a new independent batch.

## Completion Is Not Quality Approval

- **Complete / queue processed:** the committed queue finished generation and
  bounded checks/repair attempts. Some outputs can still need human review.
- **Complete / search budget exhausted:** fewer usable references were found
  than requested, but the committed usable queue finished processing.
- **Exhausted:** no references passed within the bounded search/download budget.
  A new batch can be started immediately; there is no forced resume loop.
- **Finished by user:** remaining items were deliberately left unfinished.
- **Paused / stopped:** work and files are saved and can be resumed explicitly.
- **Archived:** a reversible visibility/resume restriction, separate from status.

## Isolation and Compatibility

New batches always begin with their own search and cleaning budget. They do not
consume another batch's cached approved references, inherit another queue or
exclude a result solely because another batch used the same URL. Search services
may return the same public image again; independent search does not guarantee
globally unique results. Duplicate detection remains local to each batch.

Reference records carry a batch ID and independent validation state. Real
downloads are stored under `gallery/references/<batch-id>/`. Generated items and
derived enhancements retain their batch ID. A worker is bound to its batch for
its entire lifetime, even if the legacy active pointer changes. Job retries use
the saved batch ID; new job submissions are idempotent.

Existing batch manifests remain visible. Older shared catalogs cannot always
prove which uncommitted finding belonged to which historical search; these are
labeled legacy. On resume, matching legacy findings are copied into owned records
and committed reference IDs are remapped without deleting original files or
overwriting another batch. Confirmed identity snapshots and quality policies
remain pinned for resumed work.

Batch manifests, catalogs and results remain inside the existing results tree,
so the existing Colab Drive persistence and Kaggle Drive checkpoint sync include
them. No notebook, model, paid service or local-inference dependency is added.
