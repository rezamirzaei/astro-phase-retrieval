import { UpperCasePipe } from '@angular/common';
import { Component, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { ApiService, JobResponse } from '../core/api.service';

@Component({
  selector: 'app-result-detail',
  standalone: true,
  imports: [UpperCasePipe, RouterLink, MatCardModule, MatButtonModule, MatChipsModule],
  template: `
    @if (job()) {
      <div class="flex-row mb-16">
        <h2>{{ job()!.algorithm | uppercase }} — Result #{{ job()!.id }}</h2>
        <span class="spacer"></span>
        <button mat-stroked-button routerLink="/results">← Back</button>
        <a mat-raised-button color="primary" [href]="'/api/results/' + job()!.id + '/export'">Export ZIP</a>
      </div>
      <div class="card-grid mb-16">
        <mat-card class="stat-card"><div class="stat-value">{{ job()!.strehl_ratio !== null ? job()!.strehl_ratio!.toFixed(4) : '—' }}</div><div class="stat-label">Strehl Ratio</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ job()!.rms_phase_rad !== null ? job()!.rms_phase_rad!.toFixed(4) : '—' }}</div><div class="stat-label">RMS Phase (rad)</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ job()!.n_iterations }}</div><div class="stat-label">Iterations</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ job()!.elapsed_seconds !== null ? job()!.elapsed_seconds!.toFixed(2) : '—' }}s</div><div class="stat-label">Elapsed Time</div></mat-card>
      </div>
      <mat-chip-set class="mb-16">
        <mat-chip [highlighted]="job()!.converged === true" color="primary">{{ job()!.converged ? 'Converged ✓' : 'Did not converge' }}</mat-chip>
        <mat-chip>{{ job()!.fits_filename }}</mat-chip>
      </mat-chip-set>
      <h3>Plots</h3>
      @if (job()!.plots.length === 0) {
        <p class="text-muted">No figures were generated for this run.</p>
      }
      <div class="card-grid">
        @for (p of job()!.plots; track p) {
          <mat-card>
            @if (plotUrls()[p]) {
              <img [src]="plotUrls()[p]" [alt]="p" class="plot-img">
            } @else {
              <mat-card-content><p class="text-center text-muted">Loading figure…</p></mat-card-content>
            }
            <mat-card-content><p class="text-center text-muted">{{ p }}</p></mat-card-content>
          </mat-card>
        }
      </div>
      <h3>Artifacts</h3>
      @if (job()!.artifacts.length === 0) {
        <p class="text-muted">No saved reports were generated for this run.</p>
      } @else {
        <mat-chip-set>
          @for (artifact of job()!.artifacts; track artifact) {
            <mat-chip>{{ artifact }}</mat-chip>
          }
        </mat-chip-set>
      }
    } @else {
      <p class="text-muted">Loading…</p>
    }
  `,
})
export class ResultDetailComponent implements OnInit, OnDestroy {
  api = inject(ApiService);
  private route = inject(ActivatedRoute);
  job = signal<JobResponse | null>(null);
  plotUrls = signal<Record<string, string>>({});

  ngOnInit(): void {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    this.api.getResult(id).subscribe((j) => {
      this.job.set(j);
      this.loadPlots(j);
    });
  }

  ngOnDestroy(): void {
    this.revokePlotUrls();
  }

  private loadPlots(job: JobResponse): void {
    this.revokePlotUrls();
    for (const plotName of job.plots) {
      this.api.getPlot(job.id, plotName).subscribe({
        next: (blob) => {
          const url = URL.createObjectURL(blob);
          this.plotUrls.update((urls) => ({ ...urls, [plotName]: url }));
        },
      });
    }
  }

  private revokePlotUrls(): void {
    for (const url of Object.values(this.plotUrls())) {
      URL.revokeObjectURL(url);
    }
    this.plotUrls.set({});
  }
}


