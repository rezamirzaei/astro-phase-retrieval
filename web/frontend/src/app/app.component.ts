import { Component, inject } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { AuthService } from './core/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet, RouterLink, RouterLinkActive,
    MatToolbarModule, MatButtonModule, MatIconModule, MatSidenavModule, MatListModule,
  ],
  template: `
    <mat-toolbar color="primary" class="toolbar">
      <mat-icon>telescope</mat-icon>
      <span class="toolbar-title">Phase Retrieval</span>
      <span class="spacer"></span>
      @if (auth.isLoggedIn()) {
        <button mat-button routerLink="/dashboard" routerLinkActive="active"><mat-icon>dashboard</mat-icon> Dashboard</button>
        <button mat-button routerLink="/run" routerLinkActive="active"><mat-icon>play_arrow</mat-icon> Run</button>
        <button mat-button routerLink="/compare" routerLinkActive="active"><mat-icon>compare</mat-icon> Compare</button>
        <button mat-button routerLink="/results" routerLinkActive="active"><mat-icon>assessment</mat-icon> Results</button>
        <button mat-button routerLink="/data" routerLinkActive="active"><mat-icon>storage</mat-icon> Data</button>
        <button mat-button routerLink="/explain" routerLinkActive="active"><mat-icon>school</mat-icon> Learn</button>
        <button mat-button (click)="logout()"><mat-icon>logout</mat-icon> Logout</button>
      } @else {
        <button mat-button routerLink="/login">Login</button>
        <button mat-raised-button color="accent" routerLink="/register">Register</button>
      }
    </mat-toolbar>
    <main class="container"><router-outlet /></main>
  `,
  styles: [`
    .toolbar { position: sticky; top: 0; z-index: 100; }
    .toolbar-title { margin-left: 8px; font-weight: 500; }
    .spacer { flex: 1 1 auto; }
    .active { border-bottom: 2px solid #fff; }
  `],
})
export class AppComponent {
  auth = inject(AuthService);
  private router = inject(Router);

  logout(): void {
    this.auth.logout();
    this.router.navigate(['/login']);
  }
}

