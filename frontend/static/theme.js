// Runs in <head> before first paint: apply saved theme (else system preference) and flag JS availability.
(function () {
  var t = null;
  try { t = localStorage.getItem("emet_lens_theme"); } catch (e) {}
  if (t !== "light" && t !== "dark") t = matchMedia("(prefers-color-scheme:light)").matches ? "light" : "dark";
  var r = document.documentElement;
  r.setAttribute("data-theme", t);
  r.classList.add("js");
})();
