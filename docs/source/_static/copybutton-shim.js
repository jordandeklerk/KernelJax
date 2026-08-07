/*
 * Two things keep sphinx-copybutton from working under sphinx-immaterial, and
 * this file fixes both. It is registered ahead of copybutton.js.
 *
 * 1. copybutton.js reads DOCUMENTATION_OPTIONS.URL_ROOT at its top level. The
 *    basic theme supplies that global through documentation_options.js, which
 *    sphinx-immaterial does not emit, so the read throws a ReferenceError, the
 *    whole script aborts, and no copy button is attached anywhere. The value is
 *    only consulted to resolve a user-supplied copy icon and we use the built-in
 *    inline SVG, so an empty root is all copybutton needs.
 *
 * 2. The navigation.instant feature swaps page content without a reload and
 *    re-injects only scripts it has not already loaded, so copybutton.js never
 *    runs again and pages reached through the sidebar get no buttons. The theme
 *    publishes a document$ observable that emits on every render, instant ones
 *    included, so we attach to whatever code cells arrive without one.
 *
 * ClipboardJS is constructed against the '.copybtn' selector, which it binds by
 * delegation, so buttons added here are picked up by the instance copybutton.js
 * already created. Creating another one would copy twice per click.
 */
window.DOCUMENTATION_OPTIONS = window.DOCUMENTATION_OPTIONS || { URL_ROOT: "" };

(function () {
  var SELECTOR = "div.highlight pre";

  function icon() {
    var existing = document.querySelector("button.copybtn");
    if (existing) {
      return existing.innerHTML;
    }
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler ' +
      'icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" ' +
      'stroke-width="1.5" stroke="currentColor" fill="none" ' +
      'stroke-linecap="round" stroke-linejoin="round">' +
      '<title>Copy to clipboard</title>' +
      '<path stroke="none" d="M0 0h24v24H0z" fill="none"/>' +
      '<rect x="8" y="8" width="12" height="12" rx="2" />' +
      '<path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" />' +
      "</svg>"
    );
  }

  function attach() {
    var cells = document.querySelectorAll(SELECTOR);
    var template = null;
    for (var i = 0; i < cells.length; i++) {
      var cell = cells[i];
      var next = cell.nextElementSibling;
      if (next && next.classList && next.classList.contains("copybtn")) {
        continue;
      }
      if (!cell.id) {
        cell.setAttribute("id", "shimcodecell" + i);
      }
      if (template === null) {
        template = icon();
      }
      var button = document.createElement("button");
      button.className = "copybtn o-tooltip--left";
      button.setAttribute("data-tooltip", "Copy");
      button.setAttribute("data-clipboard-target", "#" + cell.id);
      button.innerHTML = template;
      cell.parentNode.insertBefore(button, cell.nextSibling);
    }
  }

  /*
   * copybutton.js attaches to every code cell on the page or to none, so a
   * single existing button means it has already run for this document and we
   * must keep out of its way. Without that guard the initial page would get two
   * buttons per cell.
   */
  function attachIfMissing() {
    if (document.querySelector("button.copybtn")) {
      return;
    }
    attach();
  }

  function schedule() {
    window.setTimeout(attachIfMissing, 0);
  }

  function boot() {
    /*
     * document$ is a plain subject rather than a replaying one, so a subscriber
     * registered here never sees the emission for the page it was loaded with.
     * Run once directly for that page and let the subscription cover every
     * instant navigation after it.
     */
    schedule();
    if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
      document$.subscribe(schedule);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
