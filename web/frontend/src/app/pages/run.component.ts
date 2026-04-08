import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { ApiService, AlgoDefaults, AlgoInfo, FitsFile } from '../core/api.service';

@Component({
  selector: 'app-run',
  standalone: true,
  imports: [FormsModule, MatCardModule, MatFormFieldModule, MatSelectModule, MatInputModule, MatButtonModule, MatProgressBarModule, MatSnackBarModule],
  template: `
    <h2>Run Algorithm</h2>
    <mat-card>
      <mat-card-content>
        <div class="card-grid">
          <mat-form-field>
            <mat-label>Data File</mat-label>
            <mat-select [(ngModel)]="selectedFile" name="file">
              @for (f of files(); track f.filename) { <mat-option [value]="f.filename">{{ f.filename }}</mat-option> }
            </mat-select>
          </mat-form-field>
          <mat-form-field>
            <mat-label>Algorithm</mat-label>
            <mat-select [(ngModel)]="selectedAlgo" name="algo" (ngModelChange)="applyDefaults($event)">
              @for (a of algos(); track a.key) { <mat-option [value]="a.key">{{ a.name }}</mat-option> }
            </mat-select>
          </mat-form-field>
          <mat-form-field><mat-label>Iterations</mat-label><input matInput type="number" [(ngModel)]="iterations" name="iter"></mat-form-field>
          <mat-form-field><mat-label>Grid Size</mat-label><input matInput type="number" [(ngModel)]="gridSize" name="grid"></mat-form-field>
          <mat-form-field><mat-label>β</mat-label><input matInput type="number" step="0.05" [(ngModel)]="beta" name="beta"></mat-form-field>
          <mat-form-field>
            <mat-label>β Schedule</mat-label>
            <mat-select [(ngModel)]="betaSchedule" name="bs">
              <mat-option value="constant">Constant</mat-option>
              <mat-option value="linear">Linear</mat-option>
              <mat-option value="cosine">Cosine</mat-option>
            </mat-select>
          </mat-form-field>
          <mat-form-field><mat-label>Momentum</mat-label><input matInput type="number" step="0.1" [(ngModel)]="momentum" name="mom"></mat-form-field>
          <mat-form-field><mat-label>TV Weight</mat-label><input matInput type="number" step="0.0001" [(ngModel)]="tvWeight" name="tv"></mat-form-field>
          <mat-form-field>
            <mat-label>Noise Model</mat-label>
            <mat-select [(ngModel)]="noiseModel" name="noise">
              <mat-option value="gaussian">Gaussian</mat-option>
              <mat-option value="poisson">Poisson</mat-option>
            </mat-select>
          </mat-form-field>
        </div>
        <p class="text-muted">Selecting an algorithm applies recommended defaults for that solver.</p>
      </mat-card-content>
      <mat-card-actions align="end">
        <button mat-raised-button color="primary" (click)="run()" [disabled]="loading()">
          {{ loading() ? 'Running…' : 'Run Algorithm' }}
        </button>
      </mat-card-actions>
      @if (loading()) { <mat-progress-bar mode="indeterminate"></mat-progress-bar> }
    </mat-card>
  `,
})
export class RunComponent implements OnInit {
  private api = inject(ApiService);
  private router = inject(Router);
  private snack = inject(MatSnackBar);

  files = signal<FitsFile[]>([]);
  algos = signal<AlgoInfo[]>([]);
  loading = signal(false);

  selectedFile = '';
  selectedAlgo = 'raar';
  iterations = 300;
  gridSize = 128;
  beta = 0.9;
  betaSchedule = 'constant';
  momentum = 0;
  tvWeight = 0;
  noiseModel = 'gaussian';

  ngOnInit(): void {
    this.api.getFitsFiles().subscribe(f => this.files.set(f));
    this.api.listAlgorithms().subscribe((a) => {
      this.algos.set(a);
      if (!a.some((algo) => algo.key === this.selectedAlgo) && a.length > 0) {
        this.selectedAlgo = a[0].key;
      }
      this.applyDefaults(this.selectedAlgo);
    });
  }

  applyDefaults(algorithmKey: string): void {
    const defaults = this.algos().find((algo) => algo.key === algorithmKey)?.defaults;
    if (!defaults) return;
    this.applyConfig(defaults);
  }

  private applyConfig(defaults: AlgoDefaults): void {
    this.iterations = defaults.max_iterations;
    this.gridSize = defaults.grid_size;
    this.beta = defaults.beta;
    this.betaSchedule = defaults.beta_schedule;
    this.momentum = defaults.momentum;
    this.tvWeight = defaults.tv_weight;
    this.noiseModel = defaults.noise_model;
  }

  run(): void {
    if (!this.selectedFile) { this.snack.open('Select a data file first', 'OK', { duration: 2000 }); return; }
    this.loading.set(true);
    this.api.runAlgorithm({
      fits_filename: this.selectedFile,
      algorithm: this.selectedAlgo,
      max_iterations: this.iterations,
      grid_size: this.gridSize,
      beta: this.beta,
      beta_schedule: this.betaSchedule,
      momentum: this.momentum,
      tv_weight: this.tvWeight,
      noise_model: this.noiseModel,
    }).subscribe({
      next: j => { this.loading.set(false); this.router.navigate(['/results', j.id]); },
      error: e => { this.loading.set(false); this.snack.open(e?.error?.detail || 'Run failed', 'OK', { duration: 4000 }); },
    });
  }
}

