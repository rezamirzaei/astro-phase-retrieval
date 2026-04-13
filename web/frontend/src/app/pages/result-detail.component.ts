import { JsonPipe, UpperCasePipe } from '@angular/common';
import { Component, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { ApiService, ArtifactContent, JobResponse } from '../core/api.service';

@Component({
  selector: 'app-result-detail',
  standalone: true,
  imports: [JsonPipe, UpperCasePipe, RouterLink, MatCardModule, MatButtonModule, MatChipsModule],
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

      @if (referenceValidation()) {
        <mat-card class="mb-16">
          <mat-card-header><mat-card-title>External Reference Verification</mat-card-title></mat-card-header>
          <mat-card-content>
            <p><strong>Baseline:</strong> {{ refValue('baseline', 'key') || 'unknown' }}</p>
            <p><strong>Citation:</strong> {{ refValue('baseline', 'citation_title') || 'n/a' }}</p>
            <p><strong>Observed FWHM:</strong> {{ refValue('observed', 'fwhm_arcsec') ?? 'n/a' }}</p>
            <p><strong>Reconstructed FWHM:</strong> {{ refValue('reconstructed', 'fwhm_arcsec') ?? 'n/a' }}</p>
            <p><strong>Agreement:</strong> {{ refValue('summary', 'fwhm_agreement') || 'n/a' }}</p>
          </mat-card-content>
        </mat-card>
      }

      @if (evaluationReport()) {
        <mat-card class="mb-16">
          <mat-card-header><mat-card-title>Evaluation Evidence</mat-card-title></mat-card-header>
          <mat-card-content>
            <p><strong>Validation scope:</strong> {{ nestedValue(evaluationReport()!, ['evidence', 'validation_scope']) || 'n/a' }}</p>
            <p><strong>Data regime:</strong> {{ nestedValue(evaluationReport()!, ['data_regime']) || 'n/a' }}</p>
            <p><strong>Validated claims:</strong></p>
            <ul>
              @for (claim of listValue(evaluationReport()!, 'validated_claims'); track claim) {
                <li>{{ claim }}</li>
              }
            </ul>
            <p><strong>Limitations:</strong></p>
            <ul>
              @for (limitation of listValue(evaluationReport()!, 'limitations'); track limitation) {
                <li>{{ limitation }}</li>
              }
            </ul>
          </mat-card-content>
        </mat-card>
      }

      @if (provenance()) {
        <mat-card class="mb-16">
          <mat-card-header><mat-card-title>Preprocessing Provenance</mat-card-title></mat-card-header>
          <mat-card-content>
            <pre>{{ provenance() | json }}</pre>
          </mat-card-content>
        </mat-card>
      }

      <h3>Plots</h3>
      @if (job()!.plots.length === 0) {
        <p class="text-muted">No figures were generated for this run.</p>
      }
      <div class="card-grid">
        @for (p of job()!.plots; track p) {
          <mat-card>
            @if (plotUrls()[p]) {
              <button class="plot-button" type="button" (click)="openZoom(p)">
                <img [src]="plotUrls()[p]" [alt]="p" class="plot-img">
              </button>
            } @else {
              <mat-card-content><p class="text-center text-muted">Loading figure…</p></mat-card-content>
            }
            <mat-card-content><p class="text-center text-muted">{{ p }}</p></mat-card-content>
          </mat-card>
        }
      </div>

      @if (zoomedPlotName() && zoomedPlotUrl()) {
        <div class="zoom-overlay" (click)="closeZoom()">
          <div class="zoom-dialog" (click)="$event.stopPropagation()">
            <div class="zoom-header">
              <strong>{{ zoomedPlotName() }}</strong>
              <button mat-stroked-button type="button" (click)="closeZoom()">Close</button>
            </div>
            <img [src]="zoomedPlotUrl()!" [alt]="zoomedPlotName()!" class="zoom-img">
          </div>
        </div>
      }

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
  styles: [`
    .plot-button {
      border: 0;
      background: transparent;
      padding: 0;
      width: 100%;
      cursor: zoom-in;
    }
    .zoom-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      display: grid;
      place-items: center;
      padding: 24px;
      z-index: 1000;
    }
    .zoom-dialog {
      background: #fff;
      border-radius: 8px;
      max-width: min(1200px, 100%);
      max-height: 100%;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .zoom-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .zoom-img {
      max-width: 100%;
      max-height: calc(100vh - 160px);
      object-fit: contain;
    }
  `],
})
export class ResultDetailComponent implements OnInit, OnDestroy {
  api = inject(ApiService);
  private route = inject(ActivatedRoute);
  job = signal<JobResponse | null>(null);
  plotUrls = signal<Record<string, string>>({});
  zoomedPlotName = signal<string | null>(null);
  evaluationReport = signal<Record<string, unknown> | null>(null);
  referenceValidation = signal<Record<string, unknown> | null>(null);
  provenance = signal<Record<string, unknown> | null>(null);

  ngOnInit(): void {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    this.api.getResult(id).subscribe((j) => {
      this.job.set(j);
      this.loadPlots(j);
      this.loadArtifacts(j);
    });
  }

  ngOnDestroy(): void {
    this.revokePlotUrls();
  }

  refValue(section: string, key: string): unknown {
    const data = this.referenceValidation();
    if (!data) return null;
    const sectionValue = data[section];
    if (!sectionValue || typeof sectionValue !== 'object') return null;
    return (sectionValue as Record<string, unknown>)[key] ?? null;
  }

  nestedValue(data: Record<string, unknown>, path: string[]): unknown {
    let current: unknown = data;
    for (const key of path) {
      if (!current || typeof current !== 'object') return null;
      current = (current as Record<string, unknown>)[key];
    }
    return current ?? null;
  }

  listValue(data: Record<string, unknown>, key: string): string[] {
    const value = data[key];
    return Array.isArray(value) ? value.map(item => String(item)) : [];
  }

  zoomedPlotUrl(): string | null {
    const plotName = this.zoomedPlotName();
    return plotName ? this.plotUrls()[plotName] ?? null : null;
  }

  openZoom(plotName: string): void {
    this.zoomedPlotName.set(plotName);
  }

  closeZoom(): void {
    this.zoomedPlotName.set(null);
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

  private loadArtifacts(job: JobResponse): void {
    for (const artifactName of job.artifacts) {
      this.api.getArtifactContent(job.id, artifactName).subscribe({
        next: (artifact) => this.assignArtifact(artifact),
      });
    }
  }

  private assignArtifact(artifact: ArtifactContent): void {
    if (artifact.name === 'evaluation_report.json' && artifact.format === 'json' && typeof artifact.content === 'object' && artifact.content !== null) {
      this.evaluationReport.set(artifact.content as Record<string, unknown>);
    }
    if (artifact.name === 'reference_validation.json' && artifact.format === 'json' && typeof artifact.content === 'object' && artifact.content !== null) {
      this.referenceValidation.set(artifact.content as Record<string, unknown>);
    }
    if (artifact.name === 'provenance.json' && artifact.format === 'json' && typeof artifact.content === 'object' && artifact.content !== null) {
      this.provenance.set(artifact.content as Record<string, unknown>);
    }
  }

  private revokePlotUrls(): void {
    this.closeZoom();
    for (const url of Object.values(this.plotUrls())) {
      URL.revokeObjectURL(url);
    }
    this.plotUrls.set({});
  }
}
