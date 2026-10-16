/* ==========================================================================
   PlaceScope — app shell behaviour
   Rail navigation, command palette, reveal, count-up, readiness simulator.
   ========================================================================== */
(function () {
   "use strict";

   var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
   var body = document.body;

   /* ---------------------------------------------------------------
      1. Sidebar rail: collapse (desktop) / off-canvas (mobile)
      --------------------------------------------------------------- */
   var COLLAPSE_KEY = "ps-rail-collapsed";

   try {
      if (localStorage.getItem(COLLAPSE_KEY) === "1" && window.innerWidth > 760) {
         body.classList.add("rail-collapsed");
      }
   } catch (e) { /* storage blocked — ignore */ }

   function toggleRail() {
      if (window.innerWidth <= 760) {
         body.classList.toggle("rail-open");
      } else {
         body.classList.toggle("rail-collapsed");
         try {
            localStorage.setItem(COLLAPSE_KEY, body.classList.contains("rail-collapsed") ? "1" : "0");
         } catch (e) { /* ignore */ }
      }
   }

   var railToggle = document.getElementById("railToggle");
   var tbMenu = document.getElementById("tbMenu");
   var scrim = document.getElementById("scrim");
   if (railToggle) railToggle.addEventListener("click", toggleRail);
   if (tbMenu) tbMenu.addEventListener("click", function () { body.classList.add("rail-open"); });
   if (scrim) scrim.addEventListener("click", function () { body.classList.remove("rail-open"); });

   /* ---------------------------------------------------------------
      2. Command palette (Ctrl/Cmd + K)
      --------------------------------------------------------------- */
   var cmd = document.getElementById("cmd");
   var cmdInput = document.getElementById("cmdInput");
   var cmdList = document.getElementById("cmdList");
   var cmdOpenBtn = document.getElementById("cmdOpen");

   function cmdItems() {
      return cmdList ? Array.prototype.slice.call(cmdList.querySelectorAll(".cmd-item")) : [];
   }

   function filterCmd(term) {
      var t = term.trim().toLowerCase();
      var shown = 0;
      cmdItems().forEach(function (item) {
         var text = (item.getAttribute("data-text") || "").toLowerCase();
         var match = !t || text.indexOf(t) !== -1;
         item.style.display = match ? "" : "none";
         item.classList.remove("sel");
         if (match) shown++;
      });
      var empty = document.getElementById("cmdEmpty");
      if (empty) empty.style.display = shown ? "none" : "block";
      var first = cmdItems().filter(function (i) { return i.style.display !== "none"; })[0];
      if (first) first.classList.add("sel");
   }

   function openCmd() {
      if (!cmd) return;
      cmd.classList.add("open");
      body.classList.remove("rail-open");
      if (cmdInput) { cmdInput.value = ""; filterCmd(""); }
      setTimeout(function () { if (cmdInput) cmdInput.focus(); }, 30);
   }
   function closeCmd() { if (cmd) cmd.classList.remove("open"); }

   if (cmdOpenBtn) cmdOpenBtn.addEventListener("click", openCmd);
   if (cmd) cmd.addEventListener("click", function (ev) { if (ev.target === cmd) closeCmd(); });
   if (cmdInput) cmdInput.addEventListener("input", function () { filterCmd(cmdInput.value); });

   document.addEventListener("keydown", function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
         ev.preventDefault();
         cmd && cmd.classList.contains("open") ? closeCmd() : openCmd();
      } else if (ev.key === "Escape") {
         closeCmd();
      } else if (ev.key === "Enter" && cmd && cmd.classList.contains("open")) {
         var sel = cmdList ? cmdList.querySelector(".cmd-item.sel") : null;
         if (sel) window.location.href = sel.getAttribute("href");
      } else if (cmd && cmd.classList.contains("open") && (ev.key === "ArrowDown" || ev.key === "ArrowUp")) {
         ev.preventDefault();
         var vis = cmdItems().filter(function (i) { return i.style.display !== "none"; });
         var idx = vis.findIndex(function (i) { return i.classList.contains("sel"); });
         idx = ev.key === "ArrowDown" ? Math.min(idx + 1, vis.length - 1) : Math.max(idx - 1, 0);
         vis.forEach(function (i) { i.classList.remove("sel"); });
         if (vis[idx]) { vis[idx].classList.add("sel"); vis[idx].scrollIntoView({ block: "nearest" }); }
      }
   });

   /* ---------------------------------------------------------------
      3. Scroll reveal
      --------------------------------------------------------------- */
   var revealEls = document.querySelectorAll(".reveal");
   if (reduce || !("IntersectionObserver" in window)) {
      revealEls.forEach(function (el) { el.classList.add("in"); });
   } else {
      var io = new IntersectionObserver(function (entries) {
         entries.forEach(function (entry) {
            if (entry.isIntersecting) { entry.target.classList.add("in"); io.unobserve(entry.target); }
         });
      }, { threshold: 0.12 });
      revealEls.forEach(function (el) { io.observe(el); });
   }

   /* ---------------------------------------------------------------
      4. Count-up numbers  [data-count]
      --------------------------------------------------------------- */
   function fmtNum(v, decimals, suffix) {
      return v.toLocaleString(undefined, {
         minimumFractionDigits: decimals, maximumFractionDigits: decimals
      }) + (suffix || "");
   }

   function countUp(el) {
      var target = parseFloat(el.getAttribute("data-count"));
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      var suffix = el.getAttribute("data-suffix") || "";
      if (isNaN(target)) return;
      if (reduce) { el.textContent = fmtNum(target, decimals, suffix); return; }
      var duration = 1400, start = null;
      function tick(t) {
         if (!start) start = t;
         var p = Math.min(1, (t - start) / duration);
         var eased = 1 - Math.pow(1 - p, 3);
         el.textContent = fmtNum(target * eased, decimals, suffix);
         if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
   }

   var counters = document.querySelectorAll("[data-count]");
   if (reduce || !("IntersectionObserver" in window)) {
      counters.forEach(countUp);
   } else {
      var co = new IntersectionObserver(function (entries) {
         entries.forEach(function (entry) {
            if (entry.isIntersecting) { countUp(entry.target); co.unobserve(entry.target); }
         });
      }, { threshold: 0.4 });
      counters.forEach(function (el) { co.observe(el); });
   }

   /* ---------------------------------------------------------------
      5. Animate signal bars into place
      --------------------------------------------------------------- */
   var bars = document.querySelectorAll(".signal-fill[data-w]");
   function fillBars() { bars.forEach(function (b) { b.style.width = b.getAttribute("data-w") + "%"; }); }
   if (reduce || !("IntersectionObserver" in window)) {
      fillBars();
   } else {
      var bo = new IntersectionObserver(function (entries) {
         entries.forEach(function (e) { if (e.isIntersecting) { fillBars(); bo.disconnect(); } });
      }, { threshold: 0.2 });
      if (bars.length) bo.observe(bars[0].closest(".signal") || bars[0]);
   }

   /* ---------------------------------------------------------------
      6. Readiness Console — live model prediction
      --------------------------------------------------------------- */
   var sim = document.getElementById("sim");
   if (sim) {
      var ringEl = sim.querySelector(".ring");
      var ringNum = sim.querySelector("[data-ring-num]");
      var verdict = sim.querySelector("[data-verdict]");
      var verdictIcon = sim.querySelector("[data-verdict-icon]");
      var verdictText = sim.querySelector("[data-verdict-text]");
      var verdictSub = sim.querySelector("[data-verdict-sub]");
      var gapBox = sim.querySelector("[data-gaps]");
      var noteEl = sim.querySelector("[data-note]");
      var timer = null;

      function gather() {
         var payload = {};
         sim.querySelectorAll("[data-field]").forEach(function (input) {
            payload[input.getAttribute("data-field")] = input.value;
         });
         return payload;
      }

      function render(data) {
         var p = Math.max(0, Math.min(100, data.probability));
         if (ringEl) ringEl.style.setProperty("--p", (p / 100).toFixed(4));
         if (ringNum) ringNum.textContent = p.toFixed(1);

         if (verdict) {
            verdict.classList.toggle("placed", !!data.placed);
            if (verdictIcon) verdictIcon.textContent = data.placed ? "▲" : "▼";
            if (verdictText) verdictText.textContent = data.placed ? "Likely to be placed" : "Placement unlikely";
            if (verdictSub) verdictSub.textContent = "Model estimate · " + p.toFixed(1) + "% placement probability";
         }
         if (gapBox && data.signals) {
            gapBox.innerHTML = data.signals.map(function (s) {
               var tone = s.tone === "good" ? "good" : "bad";
               var sign = s.delta >= 0 ? "+" : "";
               return '<div class="gap-row ' + tone + '">' +
                  '<span class="gr-name">' + s.label + ' vs placed median ' + s.median + '</span>' +
                  '<span class="gr-val">' + sign + s.delta + '</span></div>';
            }).join("");
         }
         if (noteEl && data.note) noteEl.textContent = data.note;
      }

      function run() {
         var payload = gather();
         noteEl && (noteEl.textContent = "Scoring…");
         fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
         })
         .then(function (r) { return r.json(); })
         .then(function (data) {
            if (data.error) { noteEl && (noteEl.textContent = "Error: " + data.error); return; }
            render(data);
         })
         .catch(function () { noteEl && (noteEl.textContent = "Could not reach the model."); });
      }

      sim.querySelectorAll("[data-field]").forEach(function (input) {
         var out = input.parentElement.querySelector("output");
         function sync() { if (out) out.textContent = input.value; }
         sync();
         input.addEventListener("input", function () {
            sync();
            clearTimeout(timer);
            timer = setTimeout(run, 220);
         });
      });

      var runBtn = sim.querySelector("[data-run]");
      if (runBtn) runBtn.addEventListener("click", run);

      run();
   }
})();
