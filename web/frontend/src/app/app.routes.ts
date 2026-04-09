import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'login',    loadComponent: () => import('./pages/login.component').then(m => m.LoginComponent) },
  { path: 'register', loadComponent: () => import('./pages/register.component').then(m => m.RegisterComponent) },
  { path: 'dashboard', canActivate: [authGuard], loadComponent: () => import('./pages/dashboard.component').then(m => m.DashboardComponent) },
  { path: 'run',       canActivate: [authGuard], loadComponent: () => import('./pages/run.component').then(m => m.RunComponent) },
  { path: 'compare',   canActivate: [authGuard], loadComponent: () => import('./pages/compare.component').then(m => m.CompareComponent) },
  { path: 'results',   canActivate: [authGuard], loadComponent: () => import('./pages/results.component').then(m => m.ResultsComponent) },
  { path: 'results/:id', canActivate: [authGuard], loadComponent: () => import('./pages/result-detail.component').then(m => m.ResultDetailComponent) },
  { path: 'data',      canActivate: [authGuard], loadComponent: () => import('./pages/data.component').then(m => m.DataComponent) },
  { path: 'crystallography', canActivate: [authGuard], loadComponent: () => import('./pages/crystallography.component').then(m => m.CrystallographyComponent) },
  { path: 'explain',   canActivate: [authGuard], loadComponent: () => import('./pages/explain.component').then(m => m.ExplainComponent) },
  { path: '**', redirectTo: 'dashboard' },
];

