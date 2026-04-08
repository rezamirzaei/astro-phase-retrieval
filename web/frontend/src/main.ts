import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { appConfig } from './app/app.config';

bootstrapApplication(AppComponent, appConfig).catch((err) => {
  console.error(err);
  // Show the error in the DOM so blank-page failures are visible
  const root = document.querySelector('app-root');
  if (root) {
    root.innerHTML = `<pre style="color:red;padding:24px;">Bootstrap failed:\n${err}</pre>`;
  }
});

