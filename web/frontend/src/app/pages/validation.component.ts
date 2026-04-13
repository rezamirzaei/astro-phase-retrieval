import { JsonPipe, UpperCasePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTableModule } from '@angular/material/table';
import { AlgoInfo, ApiService, FitsFile, ValidationCampaignResponse } from '../core/api.service';

@Component({
  selector: 'app-validation',
  standalone: true,
  imports: [JsonPipe, UpperCasePipe, FormsModule, MatButtonModule, MatCardModule, MatFormFieldModule, MatInputModule, MatProgressBarModule, MatSelectModule, MatSnackBarModule, MatTableModule],
  template: `
    <h2>Real-Data Validation Campaign</h2>
    <p class="text-muted mb-16">
      Run one algorithm across multiple real observations and inspect cross-observation consistency,
      external baseline coverage, and pass rate in one evidence-focused view.
    </p>

    <mat-card class="mb-16">
      <mat-card-content class="flex-row flex-wrap">
        <mat-form-field>
          <mat-label>Real FITS observations</mat-label>
          <mat-select [(ngModel)]="selectedFiles" name="files" multiple>
            @for (file of files(); track file.filename) {
              <mat-option [value]="file.filename">{{ file.filename }}</mat-option>
            }
          </mat-select>
        </mat-form-field>
        <mat-form-field>
          <mat-label>Algorithm</mat-label>
          <mat-select [(ngModel)]="selectedAlgorithm" name="algorithm">
            @for (algo of algorithms(); track algo.key) {
              <mat-option [value]="algo.key">{{ algo.name }}</mat-option>
            }
          </mat-select>
        </mat-form-field>
        <mat-form-field><mat-label>Iterations</mat-label><input matInput type="number" [(ngModel)]="maxIterations" name="iterations"></mat-form-field>
        <mat-form-field><mat-label>Grid Size</mat-label><input matInput type="number" [(ngModel)]="gridSize" name="grid"></mat-form-field>
        <button mat-raised-button color="primary" (click)="runCampaign()" [disabled]="loading()">
          {{ loading() ? 'Running…' : 'Run Validation Campaign' }}
        </button>
      </mat-card-content>
      @if (loading()) { <mat-progress-bar mode="indeterminate"></mat-progress-bar> }
    </mat-card>

    @if (campaign()) {
      <div class="card-grid mb-16">
        <mat-card class="stat-card"><div class="stat-value">{{ metric('n_observations') }}</div><div class="stat-label">Observations</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ percent('success_rate') }}</div><div class="stat-label">Success Rate</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ metric('reference_coverage') }}</div><div class="stat-label">Baseline Coverage</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ percent('reference_pass_rate') }}</div><div class="stat-label">Reference Pass Rate</div></mat-card>
      </div>

      <mat-card class="mb-16">
        <mat-card-header><mat-card-title>Observation Records</mat-card-title></mat-card-header>
        <mat-card-content>
          <table mat-table [dataSource]="campaign()!.records" class="mat-elevation-z1">
            <ng-container matColumnDef="source_name"><th mat-header-cell *matHeaderCellDef>Observation</th><td mat-cell *matCellDef="let row">{{ row['source_name'] }}</td></ng-container>
            <ng-container matColumnDef="filter_name"><th mat-header-cell *matHeaderCellDef>Filter</th><td mat-cell *matCellDef="let row">{{ row['filter_name'] }}</td></ng-container>
            <ng-container matColumnDef="strehl_ratio"><th mat-header-cell *matHeaderCellDef>Strehl</th><td mat-cell *matCellDef="let row">{{ formatNumber(row['strehl_ratio']) }}</td></ng-container>
            <ng-container matColumnDef="ssim"><th mat-header-cell *matHeaderCellDef>SSIM</th><td mat-cell *matCellDef="let row">{{ formatNumber(row['ssim']) }}</td></ng-container>
            <ng-container matColumnDef="reference_pass"><th mat-header-cell *matHeaderCellDef>Reference</th><td mat-cell *matCellDef="let row">{{ row['reference_available'] ? (row['reference_pass'] ? 'pass' : 'review') : 'n/a' }}</td></ng-container>
            <tr mat-header-row *matHeaderRowDef="columns"></tr>
            <tr mat-row *matRowDef="let row; columns: columns;"></tr>
          </table>
        </mat-card-content>
      </mat-card>

      <mat-card class="mb-16">
        <mat-card-header><mat-card-title>Cross-Observation Consistency</mat-card-title></mat-card-header>
        <mat-card-content><pre>{{ campaign()!.consistency | json }}</pre></mat-card-content>
      </mat-card>

      <mat-card>
        <mat-card-header><mat-card-title>Saved Artifacts</mat-card-title></mat-card-header>
        <mat-card-content><pre>{{ campaign()!.artifacts | json }}</pre></mat-card-content>
      </mat-card>
    }
  `,
})
export class ValidationComponent implements OnInit {
  private api = inject(ApiService);
  private snack = inject(MatSnackBar);
  files = signal<FitsFile[]>([]);
  algorithms = signal<AlgoInfo[]>([]);
  campaign = signal<ValidationCampaignResponse | null>(null);
  loading = signal(false);
  selectedFiles: string[] = [];
  selectedAlgorithm = 'raar';
  maxIterations = 120;
  gridSize = 128;
  columns = ['source_name', 'filter_name', 'strehl_ratio', 'ssim', 'reference_pass'];

  ngOnInit(): void {
    this.api.getFitsFiles().subscribe(files => this.files.set(files.filter(file => file.filename.endsWith('.fits'))));
    this.api.listAlgorithms().subscribe(algorithms => this.algorithms.set(algorithms));
  }

  runCampaign(): void {
    this.loading.set(true);
    this.api.runValidationCampaign({
      fits_filenames: this.selectedFiles.length > 0 ? this.selectedFiles : null,
      algorithm: this.selectedAlgorithm,
      max_iterations: this.maxIterations,
      grid_size: this.gridSize,
    }).subscribe({
      next: payload => {
        this.loading.set(false);
        this.campaign.set(payload);
      },
      error: error => {
        this.loading.set(false);
        this.snack.open(error?.error?.detail || 'Validation campaign failed', 'OK', { duration: 4000 });
      },
    });
  }

  metric(key: string): string {
    const value = this.campaign()?.summary?.[key];
    return value === undefined || value === null ? '—' : String(value);
  }

  percent(key: string): string {
    const value = this.campaign()?.summary?.[key];
    return typeof value === 'number' ? `${(value * 100).toFixed(0)}%` : '—';
  }

  formatNumber(value: unknown): string {
    return typeof value === 'number' ? value.toFixed(4) : '—';
  }
}
