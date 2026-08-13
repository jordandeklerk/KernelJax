/*
 * The theme renders the header logo as a link to the home page but leaves the
 * site title beside it as plain text. Users expect both to lead home, so this
 * wraps the title's text in an anchor pointing wherever the logo points, which
 * is already correct for the current page depth. The document$ observable
 * emits on every render including instant navigations, and the guard keeps a
 * second emission from wrapping the anchor twice.
 */
(function () {
  function linkTitle() {
    var logo = document.querySelector(".md-header__button.md-logo");
    var topic = document.querySelector(
      ".md-header__title .md-header__topic .md-ellipsis"
    );
    if (!logo || !topic || topic.closest("a")) {
      return;
    }
    var anchor = document.createElement("a");
    anchor.href = logo.getAttribute("href");
    anchor.style.color = "inherit";
    anchor.style.textDecoration = "none";
    topic.parentNode.insertBefore(anchor, topic);
    anchor.appendChild(topic);
  }

  /*
   * document$ is a plain subject rather than a replaying one, so a subscriber
   * added after the initial render misses its emission. Run once directly for
   * the current page and subscribe for the instant navigations that follow.
   */
  if (document.readyState !== "loading") {
    linkTitle();
  } else {
    document.addEventListener("DOMContentLoaded", linkTitle);
  }
  if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
    document$.subscribe(linkTitle);
  }
})();
