import { Component, inject, OnInit, signal } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatChipsModule } from '@angular/material/chips';
import { ApiService, AlgoExplain, MetricExplain } from '../core/api.service';

@Component({
  selector: 'app-explain',
  standalone: true,
  imports: [MatCardModule, MatTabsModule, MatChipsModule],
  template: `
    <h2>Learn — Phase Retrieval Science</h2>
    <mat-tab-group>
      <!-- Science -->
      <mat-tab label="The Science">
        @if (science()) {
          <div class="container">
            <h3>{{ science()!['title'] }}</h3>
            <mat-card class="mb-16"><mat-card-content><p>{{ science()!['overview'] }}</p></mat-card-content></mat-card>
            <mat-card class="mb-16"><mat-card-header><mat-card-title>The Problem</mat-card-title></mat-card-header><mat-card-content><p>{{ science()!['problem'] }}</p></mat-card-content></mat-card>
            <mat-card class="mb-16"><mat-card-header><mat-card-title>How It Works</mat-card-title></mat-card-header><mat-card-content><p>{{ science()!['method'] }}</p></mat-card-content></mat-card>
            <mat-card><mat-card-header><mat-card-title>Applications</mat-card-title></mat-card-header><mat-card-content><p style="white-space:pre-line">{{ science()!['applications'] }}</p></mat-card-content></mat-card>
          </div>
        }
      </mat-tab>
      <!-- Algorithms -->
      <mat-tab label="Algorithms">
        <div class="card-grid container">
          @for (a of algos(); track a.key) {
            <mat-card>
              <mat-card-header>
                <mat-card-title>{{ a.name }}</mat-card-title>
                <mat-card-subtitle>
                  <mat-chip-set><mat-chip>{{ a.category }}</mat-chip><mat-chip>{{ a.key }}</mat-chip></mat-chip-set>
                </mat-card-subtitle>
              </mat-card-header>
              <mat-card-content><p>{{ a.description }}</p><p class="text-muted"><em>{{ a.reference }}</em></p></mat-card-content>
            </mat-card>
          }
        </div>
      </mat-tab>
      <!-- Metrics -->
      <mat-tab label="Metrics">
        <div class="card-grid container">
          @for (m of metrics(); track m.name) {
            <mat-card>
              <mat-card-header><mat-card-title>{{ m.name }}</mat-card-title><mat-card-subtitle>{{ m.unit }}</mat-card-subtitle></mat-card-header>
              <mat-card-content><p>{{ m.description }}</p></mat-card-content>
            </mat-card>
          }
        </div>
      </mat-tab>
    </mat-tab-group>
  `,
})
export class ExplainComponent implements OnInit {
  private api = inject(ApiService);
  algos = signal<AlgoExplain[]>([]);
  metrics = signal<MetricExplain[]>([]);
  science = signal<Record<string, string> | null>(null);

  ngOnInit(): void {
    this.api.explainAlgorithms().subscribe(a => this.algos.set(a));
    this.api.explainMetrics().subscribe(m => this.metrics.set(m));
    this.api.explainScience().subscribe(s => this.science.set(s));
  }
}

