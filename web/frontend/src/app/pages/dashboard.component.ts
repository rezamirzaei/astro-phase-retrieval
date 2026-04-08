import { UpperCasePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { ApiService, DashboardStats, JobResponse } from '../core/api.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [UpperCasePipe, RouterLink, MatCardModule, MatButtonModule, MatIconModule, MatTableModule],
  template: `
    <h2>Dashboard</h2>
    @if (stats()) {
      <div class="card-grid mb-16">
        <mat-card class="stat-card"><div class="stat-value">{{ stats()!.total_runs }}</div><div class="stat-label">Total Runs</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ stats()!.completed_runs }}</div><div class="stat-label">Completed</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ stats()!.best_strehl !== null ? stats()!.best_strehl!.toFixed(4) : '—' }}</div><div class="stat-label">Best Strehl</div></mat-card>
        <mat-card class="stat-card"><div class="stat-value">{{ stats()!.algorithms_used.length }}</div><div class="stat-label">Algorithms Used</div></mat-card>
      </div>
      <div class="flex-row flex-wrap mb-16">
        <button mat-raised-button color="primary" routerLink="/run"><mat-icon>play_arrow</mat-icon> Run Algorithm</button>
        <button mat-raised-button color="accent" routerLink="/compare"><mat-icon>compare</mat-icon> Compare All</button>
        <button mat-stroked-button routerLink="/data"><mat-icon>storage</mat-icon> Manage Data</button>
        <button mat-stroked-button routerLink="/explain"><mat-icon>school</mat-icon> Learn</button>
      </div>
      @if (stats()!.recent_jobs.length > 0) {
        <h3>Recent Results</h3>
        <table mat-table [dataSource]="stats()!.recent_jobs" class="mat-elevation-z1">
          <ng-container matColumnDef="algorithm"><th mat-header-cell *matHeaderCellDef>Algorithm</th><td mat-cell *matCellDef="let j">{{ j.algorithm | uppercase }}</td></ng-container>
          <ng-container matColumnDef="status"><th mat-header-cell *matHeaderCellDef>Status</th><td mat-cell *matCellDef="let j">{{ j.status }}</td></ng-container>
          <ng-container matColumnDef="strehl"><th mat-header-cell *matHeaderCellDef>Strehl</th><td mat-cell *matCellDef="let j">{{ j.strehl_ratio !== null ? j.strehl_ratio.toFixed(4) : '—' }}</td></ng-container>
          <ng-container matColumnDef="time"><th mat-header-cell *matHeaderCellDef>Time (s)</th><td mat-cell *matCellDef="let j">{{ j.elapsed_seconds !== null ? j.elapsed_seconds.toFixed(2) : '—' }}</td></ng-container>
          <ng-container matColumnDef="actions"><th mat-header-cell *matHeaderCellDef></th><td mat-cell *matCellDef="let j"><a mat-button [routerLink]="['/results', j.id]">View</a></td></ng-container>
          <tr mat-header-row *matHeaderRowDef="displayedCols"></tr>
          <tr mat-row *matRowDef="let row; columns: displayedCols;"></tr>
        </table>
      }
    } @else {
      <p class="text-muted">Loading dashboard…</p>
    }
  `,
})
export class DashboardComponent implements OnInit {
  private api = inject(ApiService);
  stats = signal<DashboardStats | null>(null);
  displayedCols = ['algorithm', 'status', 'strehl', 'time', 'actions'];
  ngOnInit(): void { this.api.getDashboard().subscribe(s => this.stats.set(s)); }
}



