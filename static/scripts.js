// scripts.js - improved with animations, filename display, button loading and robot reactions
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("file");
const fileNameBox = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const uploadForm = document.getElementById("uploadForm");
const robotSvg = document.querySelector(".robot");

// helper: show filename nicely
function setFileName(name) {
  if (!fileNameBox) return;
  fileNameBox.textContent = name;
  fileNameBox.classList.add("visible");
  // show a temporary success badge in the drop zone
  if (!document.querySelector(".drop-success")) {
    const badge = document.createElement("div");
    badge.className = "drop-success show";
    badge.textContent = "Selected";
    dropZone.appendChild(badge);
    setTimeout(() => badge.classList.remove("show"), 2200);
    setTimeout(() => badge.remove(), 2600);
  }
}

if (dropZone && fileInput) {

  // click to open file dialog
  dropZone.addEventListener("click", () => fileInput.click());

  // when drag is over
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("hover");
    // robot reacts
    if (robotSvg) robotSvg.classList.add("drop-zone-hover-robot");
  });

  // when leaving the drop area
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("hover");
    if (robotSvg) robotSvg.classList.remove("drop-zone-hover-robot");
  });

  // drop file handler
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("hover");
    if (robotSvg) robotSvg.classList.remove("drop-zone-hover-robot");

    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      setFileName(file.name);
    }
  });

  // handle file input change (when user browses)
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) setFileName(file.name);
  });
}

// Button ripple / loading on submit
if (predictBtn && uploadForm) {
  uploadForm.addEventListener("submit", (e) => {
    // show loading state on submit — let the browser submit the form normally
    predictBtn.classList.add("loading");
    predictBtn.disabled = true;
    // Graceful fallback in case server is slow — spinner visible
    // Do not prevent default because we want real form submission.
  });

  // also add click visual ripple (pure CSS-like effect via JS)
  predictBtn.addEventListener("click", (ev) => {
    const rect = predictBtn.getBoundingClientRect();
    const ripple = document.createElement("span");
    ripple.style.position = "absolute";
    ripple.style.left = (ev.clientX - rect.left) + "px";
    ripple.style.top = (ev.clientY - rect.top) + "px";
    ripple.style.width = ripple.style.height = "12px";
    ripple.style.borderRadius = "50%";
    ripple.style.transform = "translate(-50%, -50%) scale(1)";
    ripple.style.opacity = "0.9";
    ripple.style.background = "rgba(255,255,255,0.18)";
    ripple.style.pointerEvents = "none";
    ripple.style.transition = "transform .6s ease, opacity .6s ease";
    ripple.className = "ripple";
    predictBtn.appendChild(ripple);
    requestAnimationFrame(() => {
      ripple.style.transform = "translate(-50%, -50%) scale(14)";
      ripple.style.opacity = "0";
    });
    setTimeout(() => ripple.remove(), 650);
  });
}

// Accessibility: allow keyboard file choose focus states
if (dropZone) {
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
}

// Ensure any previously selected file name is visible on page load if file input already has a file (rare)
window.addEventListener("load", () => {
  if (fileInput && fileInput.files && fileInput.files[0]) {
    setFileName(fileInput.files[0].name);
  }
});
