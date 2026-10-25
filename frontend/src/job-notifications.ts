import type { Job } from "./api";

type JobState = Pick<Job, "status" | "revision">;

export class JobFailureNotifications {
  private initialized = false;
  private seen = new Map<string, JobState>();

  consume(rows: Job[]): Job[] {
    const failures: Job[] = [];
    for (const job of rows) {
      const previous = this.seen.get(job.id);
      // Polls and stream snapshots can arrive out of order.
      if (previous && job.revision <= previous.revision) continue;
      if (
        this.initialized &&
        job.status === "failed" &&
        previous?.status !== "failed"
      ) {
        failures.push(job);
      }
      this.seen.set(job.id, { status: job.status, revision: job.revision });
    }
    // The first snapshot is saved history, not a new failure notification.
    this.initialized = true;
    return failures;
  }
}
