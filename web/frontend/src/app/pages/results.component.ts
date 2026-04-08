import { UpperCasePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { ApiService, JobResponse } from '../core/api.service';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [UpperCasePipe, RouterLink, MatTableModule, MatButtonModule, MatIconModule, MatSnackBarModule],
  template: `
    <h2>Results Gallery</h2>
    @if (results().length === 0) {
      <p class="text-muted">No results yet. <a routerLink="/run">Run an algorithm</a> to see results here.</p>
    } @else {
      <table mat-table [dataSource]="results()" class="mat-elevation-z1">
        <ng-container matColumnDef="id"><th mat-header-cell *matHeaderCellDef>#</th><td mat-cell *matCellDef="let j">{{ j.id }}</td></ng-container>
        <ng-container matColumnDef="algorithm"><th mat-header-cell *matHeaderCellDef>Algorithm</th><td mat-cell *matCellDef="let j">{{ j.algorithm | uppercase }}</td></ng-container>
        <ng-container matColumnDef="data"><th mat-header-cell *matHeaderCellDef>Data</th><td mat-cell *matCellDef="let j">{{ j.fits_filename }}</td></ng-container>
        <ng-container matColumnDef="strehl"><th mat-header-cell *matHeaderCellDef>Strehl</th><td mat-cell *matCellDef="let j">{{ j.strehl_ratio !== null ? j.strehl_ratio.toFixed(4) : '—' }}</td></ng-container>
        <ng-container matColumnDef="status"><th mat-header-cell *matHeaderCellDef>Status</th><td mat-cell *matCellDef="let j">{{ j.status }}</td></ng-container>
        <ng-container matColumnDef="actions"><th mat-header-cell *matHeaderCellDef></th><td mat-cell *matCellDef="let j">
          <a mat-icon-button [routerLink]="['/results', j.id]"><mat-icon>visibility</mat-icon></a>
          <button mat-icon-button color="warn" (click)="remove(j.id)"><mat-icon>delete</mat-icon></button>
        </td></ng-container>
        <tr mat-header-row *matHeaderRowDef="cols"></tr>
        <tr mat-row *matRowDef="let row; columns: cols;"></tr>
      </table>
    }
  `,
})
export class ResultsComponent implements OnInit {
  private api = inject(ApiService);
  private snack = inject(MatSnackBar);
  results = signal<JobResponse[]>([]);
  cols = ['id', 'algorithm', 'data', 'strehl', 'status', 'actions'];

  ngOnInit(): void { this.load(); }
  load(): void { this.api.getResults().subscribe(r => this.results.set(r)); }
  remove(id: number): void {
    this.api.deleteResult(id).subscribe({ next: () => this.load(), error: () => this.snack.open('Delete failed', 'OK', { duration: 2000 }) });
  }
}



