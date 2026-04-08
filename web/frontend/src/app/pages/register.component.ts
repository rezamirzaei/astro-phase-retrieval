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
  selector: 'app-register',
  standalone: true,
  imports: [FormsModule, RouterLink, MatCardModule, MatFormFieldModule, MatInputModule, MatButtonModule, MatSnackBarModule],
  template: `
    <div class="auth-container">
      <mat-card class="auth-card">
        <mat-card-header><mat-card-title>🔭 Create Account</mat-card-title></mat-card-header>
        <mat-card-content>
          <mat-form-field class="full-width mt-16"><mat-label>Email</mat-label><input matInput [(ngModel)]="email" name="e"></mat-form-field>
          <mat-form-field class="full-width"><mat-label>Username</mat-label><input matInput [(ngModel)]="username" name="u"></mat-form-field>
          <mat-form-field class="full-width"><mat-label>Password</mat-label><input matInput type="password" [(ngModel)]="password" name="p"></mat-form-field>
        </mat-card-content>
        <mat-card-actions align="end">
          <button mat-button routerLink="/login">Already have an account?</button>
          <button mat-raised-button color="primary" (click)="register()" [disabled]="loading">Register</button>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
})
export class RegisterComponent {
  email = ''; username = ''; password = ''; loading = false;
  private auth = inject(AuthService);
  private router = inject(Router);
  private snack = inject(MatSnackBar);

  register(): void {
    this.loading = true;
    this.auth.register(this.email, this.username, this.password).subscribe({
      next: () => { this.snack.open('Account created — please sign in', 'OK', { duration: 3000 }); this.router.navigate(['/login']); },
      error: e => { this.snack.open(e?.error?.detail || 'Registration failed', 'OK', { duration: 3000 }); this.loading = false; },
    });
  }
}

