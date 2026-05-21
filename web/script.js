(function () {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const deck = document.getElementById('deck');
  const prevBtn = document.getElementById('prev');
  const nextBtn = document.getElementById('next');
  const cur = document.getElementById('cur');
  const tot = document.getElementById('tot');
  const hint = document.querySelector('.hint');

  let idx = 0;
  tot.textContent = slides.length;

  // Cualquier tecla avanza. Las teclas de navegación reales (flechas / av-re-pág)
  // tienen su propia acción para poder retroceder también con el teclado.
  const NAV_BACK = new Set(['ArrowLeft', 'ArrowUp', 'PageUp']);
  const HOME_KEYS = new Set(['Home']);
  const END_KEYS = new Set(['End']);
  // Teclas que ignoramos del todo para que no avancen sin querer.
  const IGNORE = new Set([
    'Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'NumLock', 'ScrollLock',
    'ContextMenu', 'OS', 'Dead', 'Tab', 'Escape',
  ]);

  function render() {
    slides.forEach((s, i) => s.classList.toggle('active', i === idx));
    cur.textContent = idx + 1;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === slides.length - 1;
    history.replaceState(null, '', '#' + (idx + 1));
  }

  function go(delta) {
    const next = Math.min(slides.length - 1, Math.max(0, idx + delta));
    if (next !== idx) {
      idx = next;
      render();
    }
  }

  function jumpTo(i) {
    idx = Math.min(slides.length - 1, Math.max(0, i));
    render();
  }

  prevBtn.addEventListener('click', (e) => { e.stopPropagation(); go(-1); });
  nextBtn.addEventListener('click', (e) => { e.stopPropagation(); go(1); });

  document.addEventListener('keydown', (e) => {
    if (IGNORE.has(e.key)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key.startsWith('F') && /^F\d+$/.test(e.key)) return;

    hideHint();

    if (NAV_BACK.has(e.key))  { e.preventDefault(); go(-1); return; }
    if (HOME_KEYS.has(e.key)) { e.preventDefault(); jumpTo(0); return; }
    if (END_KEYS.has(e.key))  { e.preventDefault(); jumpTo(slides.length - 1); return; }
    // Cualquier otra tecla avanza.
    e.preventDefault();
    go(1);
  });

  // Clic sobre la diapositiva: mitad izquierda retrocede, resto avanza.
  deck.addEventListener('click', (e) => {
    if (e.target.closest('.nav') || e.target.closest('a')) return;
    const x = e.clientX;
    const w = window.innerWidth;
    if (x < w * 0.30) go(-1);
    else go(1);
    hideHint();
  });

  // Escala la diapositiva de tamaño fijo para que entre en la ventana.
  function fit() {
    const SW = 1280, SH = 720;
    const scale = Math.min(window.innerWidth / SW, window.innerHeight / SH) * 0.92;
    slides.forEach((s) => { s.style.transform = `scale(${scale})`; });
  }
  window.addEventListener('resize', fit);
  fit();

  // Restaurar desde el hash si existe.
  const fromHash = parseInt(location.hash.replace('#', ''), 10);
  if (!isNaN(fromHash)) jumpTo(fromHash - 1);
  else render();

  // Ocultar la pista pasados unos segundos.
  let hintTimer = setTimeout(hideHint, 6000);
  function hideHint() {
    if (hint) hint.classList.add('hidden');
    clearTimeout(hintTimer);
  }
})();
