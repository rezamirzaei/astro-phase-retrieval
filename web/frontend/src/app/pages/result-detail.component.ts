import { UpperCasePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
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
      <div class="card-grid">
        @for (p of job()!.plots; track p) {
          <mat-card>
            <img [src]="api.plotUrl(job()!.id, p)" [alt]="p" class="plot-img">
            <mat-card-content><p class="text-center text-muted">{{ p }}</p></mat-card-content>
          </mat-card>
        }
      </div>
    } @else {
      <p class="text-muted">Loading…</p>
    }
  `,
})
export class ResultDetailComponent implements OnInit {
  api = inject(ApiService);
  private route = inject(ActivatedRoute);
  job = signal<JobResponse | null>(null);

  ngOnInit(): void {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    this.api.getResult(id).subscribe(j => this.job.set(j));
  }
}



