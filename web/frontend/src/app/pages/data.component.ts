import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatTableModule } from '@angular/material/table';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatIconModule } from '@angular/material/icon';
import { MatExpansionModule } from '@angular/material/expansion';
import { ApiService, FitsFile, Preset } from '../core/api.service';

@Component({
  selector: 'app-data',
  standalone: true,
  imports: [FormsModule, MatCardModule, MatButtonModule, MatFormFieldModule, MatInputModule, MatSelectModule, MatTableModule, MatSnackBarModule, MatIconModule, MatExpansionModule],
  template: `
    <h2>Data Manager</h2>
    <div class="card-grid mb-16">
      <!-- Generate synthetic -->
      <mat-card>
        <mat-card-header><mat-card-title>Generate Synthetic PSF</mat-card-title></mat-card-header>
        <mat-card-content>
          <mat-form-field class="full-width"><mat-label>Name</mat-label><input matInput [(ngModel)]="synthName" name="sn"></mat-form-field>
          <mat-form-field class="full-width"><mat-label>Grid Size</mat-label><input matInput type="number" [(ngModel)]="synthGrid" name="sg"></mat-form-field>
          <mat-form-field class="full-width"><mat-label>Aberration RMS</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthRms" name="sr"></mat-form-field>
          <mat-form-field class="full-width">
            <mat-label>Telescope</mat-label>
            <mat-select [(ngModel)]="synthTelescope" name="st">
              <mat-option value="hst">HST</mat-option>
              <mat-option value="jwst">JWST</mat-option>
              <mat-option value="generic_circular">Generic Circular</mat-option>
            </mat-select>
          </mat-form-field>
          <mat-expansion-panel>
            <mat-expansion-panel-header>
              <mat-panel-title>Advanced realism controls</mat-panel-title>
            </mat-expansion-panel-header>
            <div class="card-grid">
              <mat-form-field><mat-label>Zernike Terms</mat-label><input matInput type="number" [(ngModel)]="synthNZernike" name="snz"></mat-form-field>
              <mat-form-field><mat-label>Photon Count</mat-label><input matInput type="number" [(ngModel)]="synthPhotonCount" name="spc"></mat-form-field>
              <mat-form-field><mat-label>Read Noise</mat-label><input matInput type="number" step="0.0001" [(ngModel)]="synthReadNoise" name="srn"></mat-form-field>
              <mat-form-field><mat-label>Offset Row (px)</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthOffsetRow" name="sor"></mat-form-field>
              <mat-form-field><mat-label>Offset Col (px)</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthOffsetCol" name="soc"></mat-form-field>
              <mat-form-field><mat-label>Background</mat-label><input matInput type="number" step="0.000001" [(ngModel)]="synthBackground" name="sbg"></mat-form-field>
              <mat-form-field><mat-label>Bandwidth Fraction</mat-label><input matInput type="number" step="0.01" [(ngModel)]="synthBandwidth" name="sbw"></mat-form-field>
              <mat-form-field><mat-label>Spectral Samples</mat-label><input matInput type="number" [(ngModel)]="synthSpectralSamples" name="sss"></mat-form-field>
              <mat-form-field>
                <mat-label>Spectral Weighting</mat-label>
                <mat-select [(ngModel)]="synthSpectralWeighting" name="ssw">
                  <mat-option value="delta">Delta</mat-option>
                  <mat-option value="gaussian">Gaussian</mat-option>
                  <mat-option value="uniform">Uniform</mat-option>
                </mat-select>
              </mat-form-field>
              <mat-form-field><mat-label>Field Defocus (waves)</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthFieldDefocus" name="sfd"></mat-form-field>
              <mat-form-field><mat-label>Detector Blur (px)</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthDetectorSigma" name="sds"></mat-form-field>
              <mat-form-field><mat-label>Jitter Blur (px)</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthJitterSigma" name="sjs"></mat-form-field>
              <mat-form-field><mat-label>Pixel Integration Width</mat-label><input matInput type="number" step="0.1" [(ngModel)]="synthPixelIntegrationWidth" name="spiw"></mat-form-field>
              <mat-form-field><mat-label>Random Seed</mat-label><input matInput type="number" [(ngModel)]="synthSeed" name="ssd"></mat-form-field>
            </div>
          </mat-expansion-panel>
        </mat-card-content>
        <mat-card-actions><button mat-raised-button color="primary" (click)="genSynthetic()"><mat-icon>science</mat-icon> Generate</button></mat-card-actions>
      </mat-card>
      <!-- Download preset -->
      <mat-card>
        <mat-card-header><mat-card-title>Download from MAST</mat-card-title></mat-card-header>
        <mat-card-content>
          <p class="text-muted">Download real calibrated HST observations of standard stars.</p>
          @for (p of presets(); track p.key) {
            <div class="flex-row mb-16">
              <span>{{ p.description }}</span>
              <span class="spacer"></span>
              <button mat-stroked-button (click)="download(p.key)">Download</button>
            </div>
          }
        </mat-card-content>
      </mat-card>
    </div>
    <h3>Available Data Files</h3>
    @if (files().length === 0) {
      <p class="text-muted">No data files found. Generate a synthetic PSF or download a preset above.</p>
    } @else {
      <table mat-table [dataSource]="files()" class="mat-elevation-z1">
        <ng-container matColumnDef="filename"><th mat-header-cell *matHeaderCellDef>Filename</th><td mat-cell *matCellDef="let f">{{ f.filename }}</td></ng-container>
        <ng-container matColumnDef="size"><th mat-header-cell *matHeaderCellDef>Size</th><td mat-cell *matCellDef="let f">{{ (f.size_bytes / 1024).toFixed(1) }} KB</td></ng-container>
        <tr mat-header-row *matHeaderRowDef="['filename', 'size']"></tr>
        <tr mat-row *matRowDef="let row; columns: ['filename', 'size'];"></tr>
      </table>
    }
  `,
})
export class DataComponent implements OnInit {
  private api = inject(ApiService);
  private snack = inject(MatSnackBar);
  files = signal<FitsFile[]>([]);
  presets = signal<Preset[]>([]);
  synthName = 'synthetic'; synthGrid = 128; synthRms = 0.5; synthTelescope = 'hst';
  synthNZernike = 15; synthPhotonCount = 0; synthReadNoise = 0; synthOffsetRow = 0; synthOffsetCol = 0;
  synthBackground = 0; synthBandwidth = 0; synthSpectralSamples = 1; synthSpectralWeighting = 'delta';
  synthFieldDefocus = 0; synthDetectorSigma = 0; synthJitterSigma = 0; synthPixelIntegrationWidth = 1;
  synthSeed = 42;

  ngOnInit(): void { this.refresh(); this.api.getPresets().subscribe(p => this.presets.set(p)); }
  refresh(): void { this.api.getFitsFiles().subscribe(f => this.files.set(f)); }

  genSynthetic(): void {
    this.api.generateSynthetic({
      name: this.synthName,
      grid_size: this.synthGrid,
      aberration_rms: this.synthRms,
      n_zernike: this.synthNZernike,
      telescope: this.synthTelescope,
      photon_count: this.synthPhotonCount,
      read_noise_std: this.synthReadNoise,
      center_offset_row_pixels: this.synthOffsetRow,
      center_offset_col_pixels: this.synthOffsetCol,
      background_level: this.synthBackground,
      bandwidth_fraction: this.synthBandwidth,
      spectral_samples: this.synthSpectralSamples,
      spectral_weighting: this.synthSpectralWeighting,
      field_defocus_waves: this.synthFieldDefocus,
      detector_sigma_pixels: this.synthDetectorSigma,
      jitter_sigma_pixels: this.synthJitterSigma,
      pixel_integration_width: this.synthPixelIntegrationWidth,
      random_seed: this.synthSeed,
    }).subscribe({
      next: () => { this.snack.open('Synthetic PSF generated', 'OK', { duration: 2000 }); this.refresh(); },
      error: e => this.snack.open(e?.error?.detail || 'Generation failed', 'OK', { duration: 3000 }),
    });
  }

  download(key: string): void {
    this.snack.open('Downloading… this may take a minute', 'OK', { duration: 5000 });
    this.api.downloadPreset(key).subscribe({
      next: () => { this.snack.open('Download complete', 'OK', { duration: 2000 }); this.refresh(); },
      error: e => this.snack.open(e?.error?.detail || 'Download failed (network?)', 'OK', { duration: 4000 }),
    });
  }
}
