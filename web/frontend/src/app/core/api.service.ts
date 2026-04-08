import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/* --- Shared DTOs (mirror the Pydantic schemas) --- */
export interface JobResponse {
  id: number; algorithm: string; status: string; fits_filename: string;
  strehl_ratio: number | null; rms_phase_rad: number | null;
  n_iterations: number | null; elapsed_seconds: number | null;
  converged: boolean | null; created_at: string; completed_at: string | null;
  error_message: string | null; plots: string[];
}
export interface PlotReference      { job_id: number; name: string; }
export interface CompareResponse    { results: JobResponse[]; comparison_plots: PlotReference[]; }
export interface FitsFile          { filename: string; filepath: string; size_bytes: number; }
export interface Preset            { key: string; description: string; }
export interface AlgoDefaults      { max_iterations: number; beta: number; beta_schedule: string; momentum: number; tv_weight: number; noise_model: string; grid_size: number; }
export interface AlgoInfo          { key: string; name: string; defaults: AlgoDefaults; }
export interface AlgoExplain       { key: string; name: string; category: string; description: string; reference: string; }
export interface MetricExplain     { name: string; description: string; unit: string; }
export interface DashboardStats    { total_runs: number; completed_runs: number; best_strehl: number | null; algorithms_used: string[]; recent_jobs: JobResponse[]; }

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  /* Auth helpers are in AuthService */

  /* Data */
  getPresets():  Observable<Preset[]>   { return this.http.get<Preset[]>('/api/data/presets'); }
  downloadPreset(key: string): Observable<unknown> { return this.http.post(`/api/data/download/${key}`, {}); }
  getFitsFiles(): Observable<FitsFile[]> { return this.http.get<FitsFile[]>('/api/data/fits'); }
  generateSynthetic(body: { name: string; grid_size: number; aberration_rms: number; telescope: string }): Observable<FitsFile> {
    return this.http.post<FitsFile>('/api/data/synthetic', body);
  }

  /* Algorithms */
  listAlgorithms(): Observable<AlgoInfo[]> { return this.http.get<AlgoInfo[]>('/api/algorithms/'); }
  runAlgorithm(body: Record<string, unknown>): Observable<JobResponse> { return this.http.post<JobResponse>('/api/algorithms/run', body); }
  compare(body: Record<string, unknown>): Observable<CompareResponse> { return this.http.post<CompareResponse>('/api/algorithms/compare', body); }

  /* Results */
  getResults(): Observable<JobResponse[]> { return this.http.get<JobResponse[]>('/api/results/'); }
  getResult(id: number): Observable<JobResponse> { return this.http.get<JobResponse>(`/api/results/${id}`); }
  deleteResult(id: number): Observable<void> { return this.http.delete<void>(`/api/results/${id}`); }
  plotUrl(jobId: number, name: string): string { return `/api/results/${jobId}/plots/${name}`; }
  getPlot(jobId: number, name: string): Observable<Blob> { return this.http.get(`/api/results/${jobId}/plots/${name}`, { responseType: 'blob' }); }
  getDashboard(): Observable<DashboardStats> { return this.http.get<DashboardStats>('/api/results/dashboard'); }

  /* Explain */
  explainAlgorithms(): Observable<AlgoExplain[]> { return this.http.get<AlgoExplain[]>('/api/explain/algorithms'); }
  explainMetrics():    Observable<MetricExplain[]> { return this.http.get<MetricExplain[]>('/api/explain/metrics'); }
  explainScience():    Observable<Record<string, string>> { return this.http.get<Record<string, string>>('/api/explain/science'); }
}

