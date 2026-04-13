import { Component, inject, OnInit, signal } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatIconModule } from '@angular/material/icon';
import { ApiService, FitsFile, Preset } from '../core/api.service';

@Component({
  selector: 'app-data',
  standalone: true,
  imports: [MatCardModule, MatButtonModule, MatTableModule, MatSnackBarModule, MatIconModule],
  template: `
    <h2>Real Observation Manager</h2>
    <p class="text-muted mb-16">
      This view is focused on real HST/JWST observations and verification-ready presets.
      Download calibrated MAST data, then run retrieval and inspect provenance plus external baseline checks.
    </p>

    <mat-card class="mb-16">
      <mat-card-header><mat-card-title>Download Real Data from MAST</mat-card-title></mat-card-header>
      <mat-card-content>
        @for (p of presets(); track p.key) {
          <div class="preset-row">
            <div>
              <div><strong>{{ p.key }}</strong></div>
              <div class="text-muted">{{ p.description }}</div>
              <div class="text-muted">
                {{ p.verification_supported ? 'Verification-ready baseline available' : 'No curated external baseline in repo yet' }}
              </div>
            </div>
            <button mat-stroked-button (click)="download(p.key)">
              <mat-icon>download</mat-icon> Download
            </button>
          </div>
        }
      </mat-card-content>
    </mat-card>

    <h3>Available Real Observation Files</h3>
    @if (files().length === 0) {
      <p class="text-muted">No FITS observations found yet. Download a MAST preset above.</p>
    } @else {
      <table mat-table [dataSource]="files()" class="mat-elevation-z1">
        <ng-container matColumnDef="filename"><th mat-header-cell *matHeaderCellDef>Filename</th><td mat-cell *matCellDef="let f">{{ f.filename }}</td></ng-container>
        <ng-container matColumnDef="size"><th mat-header-cell *matHeaderCellDef>Size</th><td mat-cell *matCellDef="let f">{{ (f.size_bytes / 1024).toFixed(1) }} KB</td></ng-container>
        <tr mat-header-row *matHeaderRowDef="['filename', 'size']"></tr>
        <tr mat-row *matRowDef="let row; columns: ['filename', 'size'];"></tr>
      </table>
    }
  `,
  styles: [`
    .preset-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 0;
      border-bottom: 1px solid #ececec;
    }
    .preset-row:last-child { border-bottom: 0; }
  `],
})
export class DataComponent implements OnInit {
  private api = inject(ApiService);
  private snack = inject(MatSnackBar);
  files = signal<FitsFile[]>([]);
  presets = signal<Preset[]>([]);

  ngOnInit(): void {
    this.refresh();
    this.api.getPresets().subscribe(p => this.presets.set(p));
  }

  refresh(): void {
    this.api.getFitsFiles().subscribe(f => this.files.set(f.filter(file => file.filename.endsWith('.fits'))));
  }

  download(key: string): void {
    this.snack.open('Downloading real observation… this may take a minute', 'OK', { duration: 5000 });
    this.api.downloadPreset(key).subscribe({
      next: () => {
        this.snack.open('Download complete', 'OK', { duration: 2000 });
        this.refresh();
      },
      error: e => this.snack.open(e?.error?.detail || 'Download failed (network?)', 'OK', { duration: 4000 }),
    });
  }
}
