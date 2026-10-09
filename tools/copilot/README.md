# JobBot Copilot: 1-Click In-Browser Application Assistant

JobBot Copilot is an in-browser assistant that bridges the gap between headless automation and your real daily browser. It provides zero-captcha, zero-ban autofill on **Workday**, **Greenhouse**, **Lever**, **Ashby**, and custom ATS career sites.

---

## Why Use In-Browser Copilot?

- **100% CAPTCHA-Immune:** Runs inside your authentic, logged-in browser session (Chrome, Edge, Firefox, Brave) where Cloudflare Turnstile, Datadome, and Akamai recognize you as a legitimate user.
- **Instant 1-Click Autofill:** Intelligently maps candidate contact info, LinkedIn, GitHub, address, and tailored screening questions directly to the page fields.
- **Multi-Step Memory:** Preserves loaded state across Workday multi-step wizards using `sessionStorage`.
- **Review & Submit Safety:** Smoothly scrolls and highlights the form for final confirmation before submitting.

---

## Quick Setup Options

### Option 1: Bookmarklet (Zero Install)
Create a new bookmark in your browser with the following URL:

```javascript
javascript:(function(){const s=document.createElement('script');s.src='http://localhost:5000/static/jobbot_copilot.js?t='+Date.now();document.head.appendChild(s);})();
```

Whenever you are on an application page, simply click the bookmark!

### Option 2: Tampermonkey / Violentmonkey Userscript
1. Install the Tampermonkey or Violentmonkey extension in Chrome/Edge/Firefox.
2. Create a new script and paste the contents of `jobbot_copilot.js`.
3. The Copilot floating badge will appear automatically on supported ATS pages.

### Option 3: Browser Console
1. Open DevTools (`F12` or `Ctrl+Shift+I`) on any job application page.
2. Paste the contents of `jobbot_copilot.js` into the Console and press Enter.

---

## API Endpoints

- `GET http://localhost:5000/api/job/<job_id>/copilot-package`:
  Returns the complete application package with candidate details, screening answers, resume path, and cover letter text.
- `GET http://localhost:5000/api/copilot-package?url=<current_url>`:
  Auto-detects the matching job from JobBot's database based on the browser's active URL.
