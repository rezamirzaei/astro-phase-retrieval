import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

interface Token { access_token: string; token_type: string; }
interface User  { id: number; email: string; username: string; is_active: boolean; created_at: string; }

@Injectable({ providedIn: 'root' })
export class AuthService {
  private tokenKey = 'pr_token';
  user = signal<User | null>(null);

  constructor(private http: HttpClient) {
    if (this.getToken()) { this.loadUser(); }
  }

  isLoggedIn(): boolean { return !!this.getToken(); }
  getToken(): string | null { return localStorage.getItem(this.tokenKey); }

  register(email: string, username: string, password: string): Observable<User> {
    return this.http.post<User>('/api/auth/register', { email, username, password });
  }

  login(username: string, password: string): Observable<Token> {
    return this.http.post<Token>('/api/auth/login', { username, password }).pipe(
      tap(t => { localStorage.setItem(this.tokenKey, t.access_token); this.loadUser(); }),
    );
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    this.user.set(null);
  }

  private loadUser(): void {
    this.http.get<User>('/api/auth/me').subscribe({ next: u => this.user.set(u), error: () => this.logout() });
  }
}

