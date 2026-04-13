import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/* --- Shared DTOs (mirror the Pydantic schemas) --- */
export interface JobResponse {
  id: number; algorithm: string; status: string; fits_filename: string;
  strehl_ratio: number | null; rms_phase_rad: number | null;
  n_iterations: number | null; elapsed_seconds: number | null;
  converged: boolean | null; created_at: string; completed_at: string | null;
  error_message: string | null; plots: string[]; artifacts: string[];
}
export interface PlotReference      { job_id: number; name: string; }
export interface CompareResponse    { results: JobResponse[]; comparison_plots: PlotReference[]; }
export interface FitsFile          { filename: string; filepath: string; size_bytes: number; }
export interface Preset            { key: string; description: string; verification_supported?: boolean; baseline_key?: string | null; }
export interface ArtifactContent   { name: string; format: string; content: unknown; }
export interface ValidationCampaignResponse {
  campaign_id: string;
  selected_files: string[];
  summary: Record<string, unknown>;
  records: Array<Record<string, unknown>>;
  consistency: Record<string, unknown>;
  reference_summary?: Record<string, unknown>;
  artifacts: string[];
}
export interface AlgoDefaults      {
  max_iterations: number; tolerance: number; beta: number; beta_schedule: string;
  momentum: number; tv_weight: number; noise_model: string; n_starts: number;
  uncertainty_samples: number; admm_rho: number; wf_step_size: number;
  wf_spectral_init: boolean; spectral_init: boolean; regulariser: string;
  proximal_weight: number; sparsity_threshold: number; sparsity_keep_fraction: number;
  grid_size: number;
}
export interface AlgoInfo          { key: string; name: string; defaults: AlgoDefaults; }
export interface AlgoExplain       { key: string; name: string; category: string; description: string; reference: string; }
export interface MetricExplain     { name: string; description: string; unit: string; }
export interface DashboardStats    { total_runs: number; completed_runs: number; best_strehl: number | null; algorithms_used: string[]; recent_jobs: JobResponse[]; }
export interface BenchmarkCaseInfo { key: string; description: string; }
export interface BenchmarkAggregateRow {
  algorithm: string; n_cases: number; mean_score: number; mean_ssim: number;
  mean_phase_rms_error_rad: number; mean_radial_profile_error: number;
  mean_encircled_energy_error: number; mean_elapsed_seconds: number; converged_fraction: number;
}
export interface BenchmarkStudyRow {
  algorithm: string; clean_mean_score: number; stress_mean_score: number;
  robustness_drop: number; failure_rate: number; convergence_stability: number; worst_case: string;
}
export interface BenchmarkResponse {
  selected_algorithms: string[];
  selected_cases: BenchmarkCaseInfo[];
  aggregate: BenchmarkAggregateRow[];
  study: BenchmarkStudyRow[];
  records_count: number;
  artifacts: Record<string, string>;
}

/* --- Crystallography DTOs --- */
export interface CodPreset         { key: string; description: string; }
export interface CifFile           { filename: string; filepath: string; size_bytes: number; }
export interface CrystJobResponse  {
  id: number; algorithm: string; status: string; cif_filename: string;
  cod_id: string; formula: string; r_factor: number | null;
  n_iterations: number | null; elapsed_seconds: number | null;
  converged: boolean | null; created_at: string; completed_at: string | null;
  error_message: string | null; plots: string[];
}
export interface CrystCompareResponse { results: CrystJobResponse[]; }

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  /* Auth helpers are in AuthService */

  /* Data */
  getPresets():  Observable<Preset[]>   { return this.http.get<Preset[]>('/api/data/presets'); }
  downloadPreset(key: string): Observable<unknown> { return this.http.post(`/api/data/download/${key}`, {}); }
  getFitsFiles(): Observable<FitsFile[]> { return this.http.get<FitsFile[]>('/api/data/fits'); }
  generateSynthetic(body: Record<string, unknown>): Observable<FitsFile> {
    return this.http.post<FitsFile>('/api/data/synthetic', body);
  }

  /* Algorithms */
  listAlgorithms(): Observable<AlgoInfo[]> { return this.http.get<AlgoInfo[]>('/api/algorithms/'); }
  runAlgorithm(body: Record<string, unknown>): Observable<JobResponse> { return this.http.post<JobResponse>('/api/algorithms/run', body); }
  compare(body: Record<string, unknown>): Observable<CompareResponse> { return this.http.post<CompareResponse>('/api/algorithms/compare', body); }
  getBenchmarkCases(): Observable<BenchmarkCaseInfo[]> {
    return this.http.get<BenchmarkCaseInfo[]>('/api/algorithms/benchmark/cases');
  }
  runBenchmark(body: Record<string, unknown>): Observable<BenchmarkResponse> {
    return this.http.post<BenchmarkResponse>('/api/algorithms/benchmark', body);
  }

  /* Results */
  getResults(): Observable<JobResponse[]> { return this.http.get<JobResponse[]>('/api/results/'); }
  getResult(id: number): Observable<JobResponse> { return this.http.get<JobResponse>(`/api/results/${id}`); }
  deleteResult(id: number): Observable<void> { return this.http.delete<void>(`/api/results/${id}`); }
  plotUrl(jobId: number, name: string): string { return `/api/results/${jobId}/plots/${name}`; }
  getPlot(jobId: number, name: string): Observable<Blob> { return this.http.get(`/api/results/${jobId}/plots/${name}`, { responseType: 'blob' }); }
  getArtifactContent(jobId: number, name: string): Observable<ArtifactContent> {
    return this.http.get<ArtifactContent>(`/api/results/${jobId}/artifacts/${name}`);
  }
  runValidationCampaign(body: Record<string, unknown>): Observable<ValidationCampaignResponse> {
    return this.http.post<ValidationCampaignResponse>('/api/studies/validation-campaign', body);
  }
  getValidationCampaignArtifact(campaignId: string, name: string): Observable<ArtifactContent> {
    return this.http.get<ArtifactContent>(`/api/studies/validation-campaigns/${campaignId}/artifacts/${name}`);
  }
  getDashboard(): Observable<DashboardStats> { return this.http.get<DashboardStats>('/api/results/dashboard'); }

  /* Explain */
  explainAlgorithms(): Observable<AlgoExplain[]> { return this.http.get<AlgoExplain[]>('/api/explain/algorithms'); }
  explainMetrics():    Observable<MetricExplain[]> { return this.http.get<MetricExplain[]>('/api/explain/metrics'); }
  explainScience():    Observable<Record<string, string>> { return this.http.get<Record<string, string>>('/api/explain/science'); }

  /* Crystallography */
  getCodPresets(): Observable<CodPreset[]> { return this.http.get<CodPreset[]>('/api/crystallography/presets'); }
  downloadCod(key: string): Observable<unknown> { return this.http.post(`/api/crystallography/download/${key}`, {}); }
  getCifFiles(): Observable<CifFile[]> { return this.http.get<CifFile[]>('/api/crystallography/cif-files'); }
  simulateDiffraction(body: { cif_filename: string; grid_size: number }): Observable<unknown> {
    return this.http.post('/api/crystallography/simulate', body);
  }
  runCrystallography(body: Record<string, unknown>): Observable<CrystJobResponse> {
    return this.http.post<CrystJobResponse>('/api/crystallography/run', body);
  }
  compareCrystallography(body: Record<string, unknown>): Observable<CrystCompareResponse> {
    return this.http.post<CrystCompareResponse>('/api/crystallography/compare', body);
  }
  getCrystResult(id: number): Observable<CrystJobResponse> {
    return this.http.get<CrystJobResponse>(`/api/crystallography/${id}`);
  }
  crystPlotUrl(jobId: number, name: string): string {
    return `/api/crystallography/${jobId}/plots/${name}`;
  }
  deleteCrystResult(id: number): Observable<void> {
    return this.http.delete<void>(`/api/crystallography/${id}`);
  }
}
