import { Component, inject, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatChipsModule } from '@angular/material/chips';
import { MatExpansionModule } from '@angular/material/expansion';
import { ApiService, CodPreset, CifFile, CrystJobResponse } from '../core/api.service';

@Component({
  selector: 'app-crystallography',
  standalone: true,
  imports: [
    CommonModule, FormsModule,
    MatCardModule, MatButtonModule, MatSelectModule, MatInputModule,
    MatIconModule, MatProgressSpinnerModule, MatChipsModule, MatExpansionModule,
  ],
  template: `
    <h2><mat-icon>science</mat-icon> Crystallography Phase Retrieval</h2>
    <p class="subtitle">
      Download crystal structures from the Crystallography Open Database (COD),
      simulate diffraction patterns, and recover lost phases using iterative algorithms.
    </p>

    <!-- Presets -->
    <mat-card class="section-card">
      <mat-card-header><mat-card-title>COD Presets</mat-card-title></mat-card-header>
      <mat-card-content>
        <div class="preset-grid">
          @for (p of presets(); track p.key) {
            <button mat-stroked-button (click)="downloadPreset(p.key)" [disabled]="loading()">
              <mat-icon>download</mat-icon> {{ p.key }} — {{ p.description }}
            </button>
          }
        </div>
      </mat-card-content>
    </mat-card>

    <!-- CIF Files -->
    <mat-card class="section-card">
      <mat-card-header><mat-card-title>Available CIF Files</mat-card-title></mat-card-header>
      <mat-card-content>
        <button mat-raised-button color="primary" (click)="loadCifFiles()" [disabled]="loading()">
          <mat-icon>refresh</mat-icon> Refresh
        </button>
        <div class="file-list">
          @for (f of cifFiles(); track f.filename) {
            <mat-chip (click)="selectedCif.set(f.filename)">
              {{ f.filename }} ({{ (f.size_bytes / 1024).toFixed(1) }} KB)
            </mat-chip>
          }
        </div>
        @if (selectedCif()) {
          <p>Selected: <strong>{{ selectedCif() }}</strong></p>
        }
      </mat-card-content>
    </mat-card>

    <!-- Run Phase Retrieval -->
    <mat-card class="section-card">
      <mat-card-header><mat-card-title>Run Phase Retrieval</mat-card-title></mat-card-header>
      <mat-card-content>
        <div class="form-row">
          <mat-form-field>
            <mat-label>Algorithm</mat-label>
            <mat-select [(value)]="algorithm">
              <mat-option value="er">Error Reduction (ER)</mat-option>
              <mat-option value="gs">Gerchberg-Saxton (GS)</mat-option>
              <mat-option value="hio">Hybrid Input-Output (HIO)</mat-option>
              <mat-option value="raar">RAAR</mat-option>
              <mat-option value="wf">Wirtinger Flow (WF)</mat-option>
              <mat-option value="dr">Douglas-Rachford (DR)</mat-option>
              <mat-option value="admm">ADMM</mat-option>
              <mat-option value="fista">FISTA</mat-option>
              <mat-option value="sparse_pr">Sparse PR</mat-option>
            </mat-select>
          </mat-form-field>
          <mat-form-field>
            <mat-label>Grid Size</mat-label>
            <input matInput type="number" [(ngModel)]="gridSize" min="64" max="512">
          </mat-form-field>
          <mat-form-field>
            <mat-label>Max Iterations</mat-label>
            <input matInput type="number" [(ngModel)]="maxIterations" min="1" max="10000">
          </mat-form-field>
        </div>
        <button mat-raised-button color="accent" (click)="runRetrieval()"
                [disabled]="!selectedCif() || loading()">
          <mat-icon>play_arrow</mat-icon> Run
        </button>
        <button mat-raised-button color="primary" (click)="compareAlgorithms()"
                [disabled]="!selectedCif() || loading()" style="margin-left: 8px;">
          <mat-icon>compare</mat-icon> Compare All
        </button>
        @if (loading()) {
          <mat-spinner diameter="30" style="display: inline-block; margin-left: 16px;"></mat-spinner>
        }
      </mat-card-content>
    </mat-card>

    <!-- Results -->
    @if (result()) {
      <mat-card class="section-card result-card">
        <mat-card-header><mat-card-title>Result — {{ result()!.algorithm.toUpperCase() }}</mat-card-title></mat-card-header>
        <mat-card-content>
          <div class="metrics">
            <span><strong>R-factor:</strong> {{ result()!.r_factor?.toFixed(4) }}</span>
            <span><strong>Iterations:</strong> {{ result()!.n_iterations }}</span>
            <span><strong>Time:</strong> {{ result()!.elapsed_seconds?.toFixed(2) }}s</span>
            <span><strong>Converged:</strong> {{ result()!.converged ? 'Yes' : 'No' }}</span>
          </div>
          <div class="plots-grid">
            @for (plot of result()!.plots; track plot) {
              @if (plotUrls()[result()!.id + ':' + plot]) {
                <img [src]="plotUrls()[result()!.id + ':' + plot]" [alt]="plot" class="plot-img">
              } @else {
                <p class="text-muted">Loading figure…</p>
              }
            }
          </div>
        </mat-card-content>
      </mat-card>
    }

    @if (compareResults().length > 0) {
      <mat-card class="section-card">
        <mat-card-header><mat-card-title>Algorithm Comparison</mat-card-title></mat-card-header>
        <mat-card-content>
          <table class="compare-table">
            <tr>
              <th>Algorithm</th><th>R-factor</th><th>Iterations</th><th>Time (s)</th><th>Converged</th>
            </tr>
            @for (r of compareResults(); track r.id) {
              <tr>
                <td>{{ r.algorithm.toUpperCase() }}</td>
                <td>{{ r.r_factor?.toFixed(4) }}</td>
                <td>{{ r.n_iterations }}</td>
                <td>{{ r.elapsed_seconds?.toFixed(2) }}</td>
                <td>{{ r.converged ? '✓' : '✗' }}</td>
              </tr>
            }
          </table>
        </mat-card-content>
      </mat-card>
    }
  `,
  styles: [`
    h2 { display: flex; align-items: center; gap: 8px; }
    .subtitle { color: #666; margin-bottom: 16px; }
    .section-card { margin-bottom: 16px; }
    .preset-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .file-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .form-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
    .form-row mat-form-field { min-width: 180px; }
    .metrics { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
    .plots-grid { display: flex; flex-wrap: wrap; gap: 12px; }
    .plot-img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    .compare-table { width: 100%; border-collapse: collapse; }
    .compare-table th, .compare-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
    .compare-table th { background: #f5f5f5; font-weight: 600; }
    .result-card { border-left: 4px solid #4575b4; }
    .text-muted { color: #999; font-size: 0.9em; }
  `],
})
export class CrystallographyComponent implements OnDestroy {
  api = inject(ApiService);

  presets = signal<CodPreset[]>([]);
  cifFiles = signal<CifFile[]>([]);
  selectedCif = signal<string>('');
  loading = signal(false);
  result = signal<CrystJobResponse | null>(null);
  compareResults = signal<CrystJobResponse[]>([]);
  plotUrls = signal<Record<string, string>>({});

  algorithm = 'hio';
  gridSize = 128;
  maxIterations = 500;

  constructor() {
    this.api.getCodPresets().subscribe(p => this.presets.set(p));
    this.loadCifFiles();
  }

  ngOnDestroy(): void {
    this.revokePlotUrls();
  }

  loadCifFiles(): void {
    this.api.getCifFiles().subscribe(f => this.cifFiles.set(f));
  }

  downloadPreset(key: string): void {
    this.loading.set(true);
    this.api.downloadCod(key).subscribe({
      next: () => { this.loadCifFiles(); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  runRetrieval(): void {
    this.loading.set(true);
    this.result.set(null);
    this.revokePlotUrls();
    this.api.runCrystallography({
      cif_filename: this.selectedCif(),
      algorithm: this.algorithm,
      max_iterations: this.maxIterations,
      grid_size: this.gridSize,
    }).subscribe({
      next: r => { this.result.set(r); this.loadPlots(r); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  compareAlgorithms(): void {
    this.loading.set(true);
    this.compareResults.set([]);
    this.revokePlotUrls();
    this.api.compareCrystallography({
      cif_filename: this.selectedCif(),
      max_iterations: this.maxIterations,
      grid_size: this.gridSize,
    }).subscribe({
      next: r => { this.compareResults.set(r.results); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  private loadPlots(job: CrystJobResponse): void {
    for (const plotName of job.plots) {
      this.api.getCrystPlot(job.id, plotName).subscribe({
        next: (blob) => {
          const url = URL.createObjectURL(blob);
          const key = job.id + ':' + plotName;
          this.plotUrls.update(urls => ({ ...urls, [key]: url }));
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

