import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { AuthService } from '../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, RouterLink, MatCardModule, MatFormFieldModule, MatInputModule, MatButtonModule, MatSnackBarModule],
  template: `
    <div class="auth-container">
      <mat-card class="auth-card">
        <mat-card-header><mat-card-title>🔭 Sign In</mat-card-title></mat-card-header>
        <mat-card-content>
          <mat-form-field class="full-width mt-16"><mat-label>Username</mat-label><input matInput [(ngModel)]="username" name="u"></mat-form-field>
          <mat-form-field class="full-width"><mat-label>Password</mat-label><input matInput type="password" [(ngModel)]="password" name="p" (keyup.enter)="login()"></mat-form-field>
        </mat-card-content>
        <mat-card-actions align="end">
          <button mat-button routerLink="/register">Create account</button>
          <button mat-raised-button color="primary" (click)="login()" [disabled]="loading">Sign In</button>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
})
export class LoginComponent {
  username = ''; password = ''; loading = false;
  private auth = inject(AuthService);
  private router = inject(Router);
  private snack = inject(MatSnackBar);

  login(): void {
    this.loading = true;
    this.auth.login(this.username, this.password).subscribe({
      next: () => this.router.navigate(['/dashboard']),
      error: e => { this.snack.open(e?.error?.detail || 'Login failed', 'OK', { duration: 3000 }); this.loading = false; },
    });
  }
}

