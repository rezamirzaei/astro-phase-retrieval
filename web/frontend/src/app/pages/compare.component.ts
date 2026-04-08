import { UpperCasePipe } from '@angular/common';
import { Component, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { RouterLink } from '@angular/router';
import { ApiService, CompareResponse, FitsFile } from '../core/api.service';

@Component({
  selector: 'app-compare',
  standalone: true,
  imports: [UpperCasePipe, FormsModule, RouterLink, MatCardModule, MatFormFieldModule, MatSelectModule, MatInputModule, MatButtonModule, MatProgressBarModule, MatTableModule, MatSnackBarModule],
  template: `
    <h2>Compare Algorithms</h2>
    <mat-card class="mb-16">
      <mat-card-content class="flex-row flex-wrap">
        <mat-form-field>
          <mat-label>Data File</mat-label>
          <mat-select [(ngModel)]="selectedFile" name="file">
            @for (f of files(); track f.filename) { <mat-option [value]="f.filename">{{ f.filename }}</mat-option> }
          </mat-select>
        </mat-form-field>
        <mat-form-field><mat-label>Iterations</mat-label><input matInput type="number" [(ngModel)]="iterations" name="iter"></mat-form-field>
        <mat-form-field><mat-label>Grid Size</mat-label><input matInput type="number" [(ngModel)]="gridSize" name="gs"></mat-form-field>
        <button mat-raised-button color="accent" (click)="compare()" [disabled]="loading()">
          {{ loading() ? 'Comparing…' : 'Compare All' }}
        </button>
      </mat-card-content>
      @if (loading()) { <mat-progress-bar mode="indeterminate"></mat-progress-bar> }
    </mat-card>
    @if (response()) {
      <h3>Results</h3>
      <table mat-table [dataSource]="response()!.results" class="mat-elevation-z1 mb-16">
        <ng-container matColumnDef="algorithm"><th mat-header-cell *matHeaderCellDef>Algorithm</th><td mat-cell *matCellDef="let j">{{ j.algorithm | uppercase }}</td></ng-container>
        <ng-container matColumnDef="strehl"><th mat-header-cell *matHeaderCellDef>Strehl</th><td mat-cell *matCellDef="let j">{{ j.strehl_ratio !== null ? j.strehl_ratio.toFixed(4) : '—' }}</td></ng-container>
        <ng-container matColumnDef="rms"><th mat-header-cell *matHeaderCellDef>RMS (rad)</th><td mat-cell *matCellDef="let j">{{ j.rms_phase_rad !== null ? j.rms_phase_rad.toFixed(4) : '—' }}</td></ng-container>
        <ng-container matColumnDef="iter"><th mat-header-cell *matHeaderCellDef>Iterations</th><td mat-cell *matCellDef="let j">{{ j.n_iterations }}</td></ng-container>
        <ng-container matColumnDef="time"><th mat-header-cell *matHeaderCellDef>Time (s)</th><td mat-cell *matCellDef="let j">{{ j.elapsed_seconds !== null ? j.elapsed_seconds.toFixed(2) : '—' }}</td></ng-container>
        <ng-container matColumnDef="actions"><th mat-header-cell *matHeaderCellDef></th><td mat-cell *matCellDef="let j"><a mat-button [routerLink]="['/results', j.id]">View</a></td></ng-container>
        <tr mat-header-row *matHeaderRowDef="cols"></tr>
        <tr mat-row *matRowDef="let row; columns: cols;"></tr>
      </table>
      @if (response()!.comparison_plots.length > 0) {
        <h3>Comparison Figures</h3>
        <div class="card-grid">
          @for (plot of response()!.comparison_plots; track plot.name) {
            <mat-card>
              @if (comparisonPlotUrls()[plot.name]) {
                <img [src]="comparisonPlotUrls()[plot.name]" [alt]="plot.name" class="plot-img">
              } @else {
                <mat-card-content><p class="text-center text-muted">Loading figure…</p></mat-card-content>
              }
              <mat-card-content><p class="text-center text-muted">{{ plot.name }}</p></mat-card-content>
            </mat-card>
          }
        </div>
      }
    }
  `,
})
export class CompareComponent implements OnInit, OnDestroy {
  private api = inject(ApiService);
  private snack = inject(MatSnackBar);
  files = signal<FitsFile[]>([]);
  loading = signal(false);
  response = signal<CompareResponse | null>(null);
  comparisonPlotUrls = signal<Record<string, string>>({});
  selectedFile = ''; iterations = 300; gridSize = 128;
  cols = ['algorithm', 'strehl', 'rms', 'iter', 'time', 'actions'];

  ngOnInit(): void { this.api.getFitsFiles().subscribe(f => this.files.set(f)); }
  ngOnDestroy(): void { this.revokePlotUrls(); }

  compare(): void {
    if (!this.selectedFile) { this.snack.open('Select a data file', 'OK', { duration: 2000 }); return; }
    this.loading.set(true);
    this.revokePlotUrls();
    this.api.compare({ fits_filename: this.selectedFile, max_iterations: this.iterations, grid_size: this.gridSize }).subscribe({
      next: r => {
        this.loading.set(false);
        this.response.set(r);
        for (const plot of r.comparison_plots) {
          this.api.getPlot(plot.job_id, plot.name).subscribe({
            next: (blob) => {
              const url = URL.createObjectURL(blob);
              this.comparisonPlotUrls.update((urls) => ({ ...urls, [plot.name]: url }));
            },
          });
        }
      },
      error: e => { this.loading.set(false); this.snack.open(e?.error?.detail || 'Compare failed', 'OK', { duration: 4000 }); },
    });
  }

  private revokePlotUrls(): void {
    for (const url of Object.values(this.comparisonPlotUrls())) {
      URL.revokeObjectURL(url);
    }
    this.comparisonPlotUrls.set({});
  }
}



