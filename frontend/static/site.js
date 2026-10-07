// Shared chrome: nav, footer, theme toggle, scroll reveals, count-ups, live status helpers.
(function () {
  const LOGO = '<svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="11" fill="none" stroke="var(--accent)" stroke-width="3"/><circle class="iris" cx="16" cy="16" r="4" fill="var(--accent)"/></svg>';
  const ARROW = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
  window.ARROW = ARROW;
  const NAV = [["/", "Home"], ["/analyze", "Analyze"], ["/how-it-works", "How it works"], ["/tools", "Detectors"], ["/methodology", "Methodology"]];
  const path = location.pathname.replace(/\/$/, "") || "/";
  const on = h => h === "/" ? path === "/" : path === h || path.startsWith(h + "/");

  const nav = document.getElementById("site-nav");
  if (nav) {
    nav.className = "nav";
    nav.innerHTML = `<div class="wrap">
      <a class="brand" href="/">${LOGO}Emet Lens</a>
      <nav class="links" id="links" aria-label="Main">${NAV.map(([h, t]) => `<a href="${h}"${on(h) ? ' class="on" aria-current="page"' : ""}>${t}</a>`).join("")}</nav>
      <div class="spacer"></div>
      <a class="btn cta-d" href="/analyze" style="padding:8px 16px">Analyze a file</a>
      <button class="icon-btn" id="themeBtn" aria-label="Toggle light / dark theme" title="Toggle theme">
        <svg class="sun" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <svg class="moon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z"/></svg>
      </button>
      <button class="icon-btn" id="menuBtn" aria-label="Menu" aria-expanded="false"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
    </div>`;
    document.getElementById("menuBtn").onclick = e => {
      const o = document.getElementById("links").classList.toggle("open");
      e.currentTarget.setAttribute("aria-expanded", o);
    };
    document.getElementById("themeBtn").onclick = () => {
      const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("emet_lens_theme", next); } catch (e) {}
      document.dispatchEvent(new CustomEvent("themechange"));
    };
  }

  const foot = document.getElementById("site-foot");
  if (foot) {
    foot.className = "foot";
    foot.innerHTML = `<div class="wrap"><div class="cols">
      <div><a class="brand" href="/" style="padding:0">${LOGO}Emet Lens</a>
        <p style="max-width:380px;margin:14px 0 0">Deepfake and digital-forensics checks for images, video and audio, running on your own machine.</p></div>
      <div><h4>Product</h4><a href="/analyze">Analyze a file</a><a href="/how-it-works">How it works</a><a href="/tools">Detectors</a></div>
      <div><h4>Trust</h4><a href="/methodology">Methodology</a><a href="/methodology#results">Measured results</a><a href="/methodology#limits">Limitations</a><a href="/methodology#ethics">Responsible use</a></div>
      </div><div class="fine">Results are probabilistic evidence, not proof. Only analyze content you have the right to examine. Use your own face and voice, or licensed data, when making test fakes.</div></div>`;
  }

  // scroll reveals
  const els = [...document.querySelectorAll(".rv")];
  if ("IntersectionObserver" in window && els.length) {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), {threshold: .12, rootMargin: "0px 0px -40px 0px"});
    els.forEach(el => io.observe(el));
  } else els.forEach(el => el.classList.add("in"));

  // helpers
  window.countUp = function (el, end, {dur = 900, dec = 0, suffix = "", fmt = null} = {}) {
    const out = v => fmt ? fmt(v) + suffix : v.toFixed(dec) + suffix;
    if (matchMedia("(prefers-reduced-motion:reduce)").matches) { el.textContent = out(end); return; }
    const t0 = performance.now();
    (function step(t) { const k = Math.min(1, (t - t0) / dur); el.textContent = out(end * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); })(t0);
  };
  let sp = null;
  window.getStatus = () => sp || (sp = fetch("/api/status").then(r => r.json()).catch(() => null));
  window.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
})();
