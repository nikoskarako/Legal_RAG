import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import json
import re
import math
import threading
import chat_openrouter  # Your backend processing module

CATEGORY_MAP = {
    "Α.Ν.": "Αναγκαστικός Νόμος",
    "Β.Δ.": "Βασιλικό Διάταγμα",
    "Ν.": "Νόμος",
    "Ν.Δ.": "Νομοθετικό Διάταγμα",
    "Ν.Π.Δ.": "Νομικό Πρόσωπο Δημοσίου Δικαίου",
    "Π.Δ.": "Προεδρικό Διάταγμα",
    "Π.Ν.Π.": "Πράξη Νομοθετικού Περιεχομένου",
}

# ─────────────────────────────────────────
# Design Tokens
# ─────────────────────────────────────────
C = {
    "bg":          "#F0F4F8",
    "header_bg":   "#1C2B4A",
    "header_fg":   "#FFFFFF",
    "header_sub":  "#8DA6CC",
    "ctrl_bg":     "#E8EDF5",
    "border":      "#CBD5E8",
    "canvas_bg":   "#F0F4F8",
    "user_bg":     "#2C5FC4",   # deep blue
    "user_fg":     "#FFFFFF",
    "bot_bg":      "#FFFFFF",
    "bot_fg":      "#1A1A2E",
    "bot_border":  "#D0D7E8",
    "send_bg":     "#2C5FC4",
    "send_fg":     "#FFFFFF",
    "send_hover":  "#1E47A0",
    "label_user":  "#2C5FC4",
    "label_bot":   "#6B7A99",
    "input_bg":    "#FFFFFF",
    "cursor":      "#1C2B4A",
    "cite_fg":     "#2C5FC4",   # citation [S#] color
}

FONT_BODY   = ("Helvetica", 12)
FONT_BOLD   = ("Helvetica", 12, "bold")
FONT_LINK   = ("Helvetica", 12, "bold", "underline")
FONT_SMALL  = ("Helvetica", 10)
FONT_LABEL  = ("Helvetica", 11, "bold")
FONT_TITLE  = ("Helvetica", 15, "bold")
FONT_SUB    = ("Helvetica", 10)
FONT_SEND   = ("Helvetica", 12, "bold")
FONT_MONO   = ("Courier", 11)

WRAP_CHARS  = 68   # characters per line estimate for height calc

# ─────────────────────────────────────────
# Markdown → tk.Text renderer
# ─────────────────────────────────────────
def render_markdown(text_widget, raw: str, fg: str, bg: str, cite_tag: str = None):
    """Insert markdown-formatted text into a tk.Text widget using tags.

    cite_tag: optional extra tag applied to [S#] ranges, used to scope click bindings
              to a specific message (so each message's sources are isolated).
    """
    text_widget.config(state="normal")
    text_widget.delete("1.0", "end")

    # Configure tags once
    text_widget.tag_configure("bold",    font=FONT_BOLD,   foreground=fg)
    text_widget.tag_configure("italic",  font=("Helvetica", 12, "italic"), foreground=fg)
    text_widget.tag_configure("normal",  font=FONT_BODY,   foreground=fg)
    text_widget.tag_configure("cite",    font=FONT_BOLD,   foreground=C["cite_fg"] if bg != C["user_bg"] else "#B8D4FF")
    text_widget.tag_configure("mono",    font=FONT_MONO,   foreground=fg, background=bg)
    text_widget.tag_configure("bullet",  font=FONT_BODY,   foreground=fg, lmargin1=10, lmargin2=22)
    text_widget.tag_configure("heading", font=("Helvetica", 13, "bold"), foreground=fg)

    lines = raw.split("\n")
    for line in lines:
        # Blank line → paragraph gap
        if not line.strip():
            text_widget.insert("end", "\n")
            continue

        # Heading: ## or ###
        heading_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading_match:
            _insert_inline(text_widget, heading_match.group(2), "heading", cite_tag)
            text_widget.insert("end", "\n")
            continue

        # Bullet: - item or * item
        bullet_match = re.match(r'^[\*\-]\s+(.*)', line)
        if bullet_match:
            text_widget.insert("end", "  • ", "bullet")
            _insert_inline(text_widget, bullet_match.group(1), "bullet", cite_tag)
            text_widget.insert("end", "\n")
            continue

        # Numbered list: 1. item
        num_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if num_match:
            text_widget.insert("end", f"  {num_match.group(1)}. ", "normal")
            _insert_inline(text_widget, num_match.group(2), "normal", cite_tag)
            text_widget.insert("end", "\n")
            continue

        # Normal line
        _insert_inline(text_widget, line, "normal", cite_tag)
        text_widget.insert("end", "\n")

    text_widget.config(state="disabled")


def _insert_inline(widget, text: str, base_tag: str, cite_tag: str = None):
    """Parse inline bold/italic/cite markers and insert with tags."""
    # Pattern order matters: bold before italic
    pattern = re.compile(
        r'(\*\*(.+?)\*\*'                            # bold
        r'|\*(.+?)\*'                                # italic
        r'|\[\s*S\d+(?:\s*,\s*S?\d+)*\s*\]'         # citations: [S1], [S1,S2], [S1, S2, S3]
        r'|\`([^\`]+)\`'                            # inline code
        r')',
        re.DOTALL
    )
    cite_tags = ("cite", cite_tag) if cite_tag else ("cite",)
    pos = 0
    for m in pattern.finditer(text):
        # Insert plain text before this match
        if m.start() > pos:
            widget.insert("end", text[pos:m.start()], base_tag)
        raw = m.group(0)
        if raw.startswith("**"):
            widget.insert("end", m.group(2), "bold")
        elif raw.startswith("*"):
            widget.insert("end", m.group(3), "italic")
        elif raw.startswith("[S"):
            widget.insert("end", raw, cite_tags)
        elif raw.startswith("`"):
            widget.insert("end", m.group(4), "mono")
        pos = m.end()
    # Remainder
    if pos < len(text):
        widget.insert("end", text[pos:], base_tag)


def calc_display_lines(text: str) -> int:
    """
    Calculate the number of wrapped display lines using font metrics.
    Works synchronously before the widget is created — no after() needed.
    """
    font = tkfont.Font(family="Helvetica", size=12)
    # Tk Text widget 'width' is measured in units of character '0'
    ref_px      = font.measure("0")
    content_px  = max(1, WRAP_CHARS * ref_px - 2 * 12)   # 12 = padx

    total = 0
    for line in text.split("\n"):
        if not line.strip():
            total += 1
            continue
        line_px = font.measure(line)
        total  += max(1, math.ceil(line_px / content_px))
    return max(1, total)


# ─────────────────────────────────────────
# Citations Panel
# ─────────────────────────────────────────
_OPEN_LAW_WINDOWS = {}    # doc_id -> Toplevel; reuse if already open
_OPEN_CHUNK_WINDOWS = {}  # (msg_id, s_index) -> Toplevel
_MSG_COUNTER = [0]
_MSG_SOURCES = {}         # msg_id -> {s_index: source_dict}


def _next_msg_id() -> int:
    _MSG_COUNTER[0] += 1
    return _MSG_COUNTER[0]


def open_chunk_window(msg_id: int, s_index: int):
    """Pop up the retrieved chunk text for a specific [S#] in a specific message."""
    src_map = _MSG_SOURCES.get(msg_id) or {}
    src = src_map.get(s_index)
    if not src:
        return

    key = (msg_id, s_index)
    existing = _OPEN_CHUNK_WINDOWS.get(key)
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        existing.focus_force()
        return

    title_str = (src.get("nomosTitle") or src.get("label")
                 or src.get("doc_id") or f"S{s_index}").strip()
    win = tk.Toplevel()
    win.title(f"[S{s_index}] {title_str[:80]}")
    win.configure(bg=C["bg"])
    win.geometry("780x560")
    _OPEN_CHUNK_WINDOWS[key] = win

    def _cleanup():
        _OPEN_CHUNK_WINDOWS.pop(key, None)
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", _cleanup)

    # Header
    hdr = tk.Frame(win, bg=C["header_bg"], padx=14, pady=10)
    hdr.pack(fill="x")
    tk.Label(hdr, text=f"[S{s_index}]   {title_str}",
             bg=C["header_bg"], fg=C["header_fg"],
             font=FONT_LABEL, anchor="w", justify="left",
             wraplength=720).pack(fill="x")
    parts = []
    if src.get("fekDate"):
        parts.append(f"Date: {src['fekDate']}")
    cat = src.get("nomosCategory")
    if cat:
        cat_desc = CATEGORY_MAP.get(cat, "")
        parts.append(f"Type: {cat}" + (f" — {cat_desc}" if cat_desc else ""))
    if src.get("doc_id"):
        parts.append(f"doc_id: {src['doc_id']}")
    parts.append("retrieved chunk")
    tk.Label(hdr, text="   ·   ".join(parts), bg=C["header_bg"], fg=C["header_sub"],
             font=FONT_SMALL, anchor="w").pack(fill="x", pady=(2, 0))

    # Body
    body = tk.Frame(win, bg=C["bot_bg"])
    body.pack(fill="both", expand=True, padx=10, pady=10)
    txt = tk.Text(body, wrap="word", font=FONT_BODY,
                  bg=C["bot_bg"], fg=C["bot_fg"],
                  bd=0, padx=10, pady=10, state="normal")
    sb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    chunk = (src.get("chunk_text") or "").strip()
    txt.insert("1.0", chunk if chunk else "(No chunk text available.)")
    txt.config(state="disabled")


def _make_cite_click_handler(msg_id: int, cite_tag: str):
    """Return a handler that figures out which [S#] in a multi-citation was clicked
    and opens the corresponding chunk window."""
    def _on_click(event):
        widget = event.widget
        click_idx = widget.index(f"@{event.x},{event.y}")
        # Find the bracket range containing this click
        rng = widget.tag_prevrange(cite_tag, f"{click_idx}+1c")
        if not rng:
            return
        start, end = rng
        bracket = widget.get(start, end)  # e.g. "[S1]" or "[S1,S2]"
        s_matches = list(re.finditer(r"S(\d+)", bracket))
        if not s_matches:
            return
        # Click offset (in chars) relative to bracket start
        try:
            click_off = widget.count(start, click_idx, "chars")[0]
        except Exception:
            click_off = 0
        chosen = s_matches[0]
        for m in s_matches:
            if m.start() <= click_off < m.end():
                chosen = m
                break
            if abs(m.start() - click_off) < abs(chosen.start() - click_off):
                chosen = m
        open_chunk_window(msg_id, int(chosen.group(1)))
    return _on_click


def open_law_window(doc_id: str, fallback_title: str = ""):
    """Open a popup showing the full text of a law (looked up by doc_id)."""
    # Reuse existing window for the same doc_id
    existing = _OPEN_LAW_WINDOWS.get(doc_id)
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        existing.focus_force()
        return

    win = tk.Toplevel()
    win.title(fallback_title[:80] if fallback_title else f"Law {doc_id}")
    win.configure(bg=C["bg"])
    win.geometry("780x640")
    _OPEN_LAW_WINDOWS[doc_id] = win

    def _cleanup():
        _OPEN_LAW_WINDOWS.pop(doc_id, None)
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", _cleanup)

    # Header
    hdr = tk.Frame(win, bg=C["header_bg"], padx=14, pady=10)
    hdr.pack(fill="x")
    title_lbl = tk.Label(hdr, text=fallback_title or "Loading…",
                         bg=C["header_bg"], fg=C["header_fg"],
                         font=FONT_LABEL, anchor="w", justify="left",
                         wraplength=720)
    title_lbl.pack(fill="x")
    meta_lbl = tk.Label(hdr, text="", bg=C["header_bg"], fg=C["header_sub"],
                        font=FONT_SMALL, anchor="w")
    meta_lbl.pack(fill="x", pady=(2, 0))

    # Body: scrollable Text
    body = tk.Frame(win, bg=C["bot_bg"])
    body.pack(fill="both", expand=True, padx=10, pady=10)
    txt = tk.Text(body, wrap="word", font=FONT_BODY,
                  bg=C["bot_bg"], fg=C["bot_fg"],
                  bd=0, padx=10, pady=10, state="disabled")
    sb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def _set_body(message: str):
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.insert("1.0", message)
        txt.config(state="disabled")

    _set_body("Loading…")

    def _worker():
        info = chat_openrouter.lookup_full_law_text(doc_id)

        def _apply():
            if not win.winfo_exists():
                return
            if not info:
                title_lbl.config(text=fallback_title or f"Law {doc_id}")
                meta_lbl.config(text=f"doc_id: {doc_id}")
                _set_body(
                    "Full law text is not available.\n\n"
                    "Either the full-text database (HARVESTER_FULL_DB) is missing "
                    f"or no row matched doc_id={doc_id!r}."
                )
                return
            t = info.get("nomosTitle") or fallback_title or f"Law {doc_id}"
            title_lbl.config(text=t)
            parts = []
            if info.get("fekDate"):
                parts.append(f"Date: {info['fekDate']}")
            cat = info.get("nomosCategory")
            if cat:
                cat_desc = CATEGORY_MAP.get(cat, "")
                parts.append(f"Type: {cat}" + (f" — {cat_desc}" if cat_desc else ""))
            parts.append(f"doc_id: {doc_id}")
            meta_lbl.config(text="   ·   ".join(parts))
            _set_body(info.get("text") or "(No text column found in the full-text database.)")
        win.after(0, _apply)

    threading.Thread(target=_worker, daemon=True).start()


def _citation_entry(parent: tk.Frame, e: dict):
    """One structured citation row inside the citations card."""
    s_idxs    = e.get("s_indices") or []
    s_tag     = ", ".join(f"S{i}" for i in s_idxs if isinstance(i, int))
    title     = (e.get("nomosTitle") or e.get("label") or e.get("doc_id") or "Unknown").strip()
    fek_date  = (e.get("fekDate") or "").strip()
    category  = (e.get("nomosCategory") or "").strip()
    cat_desc  = CATEGORY_MAP.get(category, "") if category else ""

    row = tk.Frame(parent, bg=C["bot_bg"], padx=14, pady=10)
    row.pack(fill="x")
    row.columnconfigure(1, weight=1)

    # ── [S#] badge ──────────────────────
    if s_tag:
        badge_frame = tk.Frame(row, bg=C["user_bg"], padx=6, pady=2)
        badge_frame.grid(row=0, column=0, sticky="nw", padx=(0, 12), pady=(1, 0))
        tk.Label(badge_frame, text=f"[{s_tag}]",
                 bg=C["user_bg"], fg="#FFFFFF",
                 font=("Helvetica", 10, "bold")).pack()

    # ── Law title (clickable → opens full-text popup) ─────────
    doc_id = (e.get("doc_id") or "").strip()
    title_lbl = tk.Label(row, text=title,
                         bg=C["bot_bg"], fg=C["cite_fg"] if doc_id else C["bot_fg"],
                         font=FONT_LINK if doc_id else FONT_BOLD,
                         anchor="w", justify="left",
                         wraplength=450,
                         cursor="hand2" if doc_id else "")
    title_lbl.grid(row=0, column=1, sticky="ew")
    if doc_id:
        title_lbl.bind("<Button-1>", lambda _ev, did=doc_id, t=title: open_law_window(did, t))

    # ── Metadata pills ──────────────────
    meta = tk.Frame(row, bg=C["bot_bg"])
    meta.grid(row=1, column=1, sticky="w", pady=(5, 0))

    def pill(parent, label, value, label_fg, value_fg, value_font=FONT_SMALL):
        f = tk.Frame(parent, bg=C["ctrl_bg"], padx=7, pady=3)
        f.pack(side="left", padx=(0, 6))
        tk.Label(f, text=label, bg=C["ctrl_bg"], fg=label_fg,
                 font=("Helvetica", 9, "bold")).pack(side="left")
        tk.Label(f, text=value, bg=C["ctrl_bg"], fg=value_fg,
                 font=value_font).pack(side="left", padx=(3, 0))

    if fek_date:
        pill(meta, "Date", fek_date, C["label_bot"], C["bot_fg"])
    if category:
        cat_text = category + (f"  —  {cat_desc}" if cat_desc else "")
        pill(meta, "Type", cat_text, C["label_bot"], C["bot_fg"])


def add_citations(parsed: dict):
    """Render a styled citations card (replaces the plain markdown bot bubble)."""
    entries       = parsed.get("cited_sources_dedup") or []
    fallback_text = ""

    if not entries:
        cf = (parsed.get("citations_formatted") or "").strip()
        lines = [ln.strip().lstrip("-").strip() for ln in cf.splitlines() if ln.strip()]
        if not lines:
            return
        fallback_text = lines   # plain-text fallback

    # ── Outer wrapper (mirrors add_message layout) ──────────────────────
    outer = tk.Frame(messages_frame, bg=C["canvas_bg"])
    outer.pack(fill="x", pady=2)

    tk.Label(outer, text="Legal Assistant", bg=C["canvas_bg"], fg=C["label_bot"],
             font=FONT_SMALL).pack(anchor="w", padx=14, pady=(4, 1))

    card = tk.Frame(outer, bg=C["bot_bg"], bd=1, relief="solid",
                    highlightbackground=C["bot_border"], highlightthickness=1)
    card.pack(side="left", padx=(6, 60), pady=(0, 4))

    # ── Header bar ──────────────────────────────────────────────────────
    hdr = tk.Frame(card, bg=C["header_bg"], padx=14, pady=8)
    hdr.pack(fill="x")
    tk.Label(hdr, text="Legal Sources Referenced",
             bg=C["header_bg"], fg=C["header_fg"],
             font=FONT_LABEL).pack(side="left")
    n = len(entries) if entries else len(fallback_text)
    tk.Label(hdr, text=f"{n} document{'s' if n != 1 else ''}",
             bg=C["header_bg"], fg=C["header_sub"],
             font=FONT_SMALL).pack(side="right")

    # ── Entries ─────────────────────────────────────────────────────────
    if entries:
        for i, e in enumerate(entries):
            if i > 0:
                tk.Frame(card, bg=C["border"], height=1).pack(fill="x")
            _citation_entry(card, e)
    else:
        for line in fallback_text:
            f = tk.Frame(card, bg=C["bot_bg"], padx=14, pady=8)
            f.pack(fill="x")
            tk.Label(f, text=line, bg=C["bot_bg"], fg=C["bot_fg"],
                     font=FONT_BODY, anchor="w", wraplength=460,
                     justify="left").pack(fill="x")


# ─────────────────────────────────────────
# Chat Functions
# ─────────────────────────────────────────
def add_message(raw_text: str, sender: str, sources_numbered=None):
    is_user = sender == "user"
    bg      = C["user_bg"] if is_user else C["bot_bg"]
    fg      = C["user_fg"] if is_user else C["bot_fg"]

    outer = tk.Frame(messages_frame, bg=C["canvas_bg"])
    outer.pack(fill="x", pady=2)

    # Sender label row
    label_row = tk.Frame(outer, bg=C["canvas_bg"])
    label_row.pack(fill="x", padx=14, pady=(4, 1))
    label_text = "You" if is_user else "Legal Assistant"
    label_fg   = C["label_user"] if is_user else C["label_bot"]
    tk.Label(label_row, text=label_text, bg=C["canvas_bg"], fg=label_fg,
             font=FONT_SMALL).pack(side="right" if is_user else "left")

    # Bubble row
    bubble_row = tk.Frame(outer, bg=C["canvas_bg"])
    bubble_row.pack(fill="x", padx=10, pady=(0, 4))

    w = WRAP_CHARS

    if is_user:
        # Thin border card, right-aligned with left indent
        spacer = tk.Frame(bubble_row, bg=C["canvas_bg"])
        spacer.pack(side="left", fill="x", expand=True)

        card = tk.Frame(bubble_row, bg=bg, bd=0)
        card.pack(side="right", padx=(60, 6))
    else:
        card = tk.Frame(bubble_row, bg=bg, bd=1, relief="solid",
                        highlightbackground=C["bot_border"], highlightthickness=1)
        card.pack(side="left", padx=(6, 60))

    txt = tk.Text(
        card,
        font=FONT_BODY,
        bg=bg,
        fg=fg,
        relief="flat",
        bd=0,
        wrap="word",
        width=w,
        height=calc_display_lines(raw_text),
        cursor="arrow",
        spacing1=3, 
        spacing3=3,
        padx=12,
        pady=8,
        exportselection=True,
    )
    txt.pack(fill="x")

    # Per-message cite tag — only set if we have chunk-level sources to look up
    cite_tag = None
    if not is_user and sources_numbered:
        msg_id = _next_msg_id()
        _MSG_SOURCES[msg_id] = {
            int(s["s_index"]): s for s in sources_numbered
            if isinstance(s, dict) and isinstance(s.get("s_index"), int)
        }
        cite_tag = f"cite_m{msg_id}"

    render_markdown(txt, raw_text, fg, bg, cite_tag=cite_tag)

    if cite_tag:
        # Make [S#] feel clickable: hand cursor + underline only on this message's cites
        link_font = FONT_LINK if bg != C["user_bg"] else FONT_LINK
        txt.tag_configure(cite_tag, font=link_font)
        txt.tag_bind(cite_tag, "<Enter>", lambda _e: txt.config(cursor="hand2"))
        txt.tag_bind(cite_tag, "<Leave>", lambda _e: txt.config(cursor="arrow"))
        txt.tag_bind(cite_tag, "<Button-1>", _make_cite_click_handler(msg_id, cite_tag))

    # Instance binding fires before Text's class binding.
    # Delegate to the canvas's smooth-scroll handler and return "break" so Text doesn't scroll internally.
    txt.bind("<MouseWheel>",
             lambda e: (on_mousewheel(e), "break")[1])


def format_sources_markdown(sources, title="Sources"):
    if not sources:
        return None
    lines = []
    for i, s in enumerate(sources, start=1):
        if isinstance(s, str):
            lines.append(f"- {i}. {s}")
            continue
        title_txt = (s.get("label") or s.get("title") or s.get("doc_id") or s.get("url") or "Source")
        url = s.get("url")
        lines.append(f"{i}. **{title_txt}**" + (f" ({url})" if url else ""))
    return f"**{title}**\n" + "\n".join(lines)


def format_legend_markdown(parsed: dict, title="Legend (inline [S#] → document)"):
    if not isinstance(parsed, dict):
        return None
    numbered = parsed.get("sources_numbered") or []
    if not numbered:
        return None
    lines = []
    for it in numbered:
        s = it.get("s_index")
        doc_id = it.get("doc_id")
        if not isinstance(s, int) or not doc_id:
            continue
        lines.append(f"- **[S{s}]** → {doc_id}")
    if not lines:
        return None
    return f"**{title}**\n" + "\n".join(lines)


def format_citations_markdown(parsed: dict, title="Citations"):
    if not isinstance(parsed, dict):
        return None

    cf = parsed.get("citations_formatted")
    if cf:
        cf = str(cf).strip()
        if cf:
            return f"**{title}**\n" + "\n".join([f"- {ln}" for ln in cf.splitlines() if ln.strip()])

    entries = parsed.get("cited_sources_dedup") or []
    if not entries:
        return None

    lines = []
    for e in entries:
        s_idxs = e.get("s_indices") or []
        s_tag = ",".join([f"S{i}" for i in s_idxs if isinstance(i, int)])
        s_tag = f"[{s_tag}]" if s_tag else ""

        nomos_title = (e.get("nomosTitle") or e.get("label") or e.get("doc_id") or "").strip()
        fek_date    = e.get("fekDate")
        category    = (e.get("nomosCategory") or "").strip()

        cat_desc = CATEGORY_MAP.get(category, "") if category else ""
        cat_str  = f"{category}" + (f" ({cat_desc})" if cat_desc else "") if category else ""

        parts = []
        if nomos_title: parts.append(nomos_title)
        if fek_date:    parts.append(f"FEK date: {fek_date}")
        if cat_str:     parts.append(f"Category: {cat_str}")

        label = " — ".join(parts) if parts else (e.get("doc_id") or "Source")
        lines.append(f"- **{s_tag}** → {label}" if s_tag else f"- {label}")

    return f"**{title}**\n" + "\n".join(lines)


# ─────────────────────────────────────────
# Typing Indicator
# ─────────────────────────────────────────
_typing_frame    = None
_typing_label    = None
_typing_anim_job = None
_typing_dots     = 0

def _show_typing():
    global _typing_frame, _typing_label, _typing_dots

    _typing_frame = tk.Frame(messages_frame, bg=C["canvas_bg"])
    _typing_frame.pack(fill="x", pady=2)

    tk.Label(_typing_frame, text="Legal Assistant",
             bg=C["canvas_bg"], fg=C["label_bot"],
             font=FONT_SMALL).pack(anchor="w", padx=14, pady=(4, 1))

    bubble_row = tk.Frame(_typing_frame, bg=C["canvas_bg"])
    bubble_row.pack(fill="x", padx=10, pady=(0, 4))

    card = tk.Frame(bubble_row, bg=C["bot_bg"], bd=1, relief="solid",
                    highlightbackground=C["bot_border"], highlightthickness=1)
    card.pack(side="left", padx=(6, 60))

    _typing_label = tk.Label(card, text="Thinking",
                              bg=C["bot_bg"], fg=C["label_bot"],
                              font=("Helvetica", 12, "italic"),
                              padx=18, pady=12)
    _typing_label.pack()

    _typing_dots = 0
    _tick_typing()

    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)


def _tick_typing():
    global _typing_anim_job, _typing_dots
    if _typing_label is None:
        return
    _typing_label.config(text="Thinking" + "." * (_typing_dots % 4))
    _typing_dots += 1
    _typing_anim_job = root.after(420, _tick_typing)


def _hide_typing():
    global _typing_frame, _typing_label, _typing_anim_job
    if _typing_anim_job:
        root.after_cancel(_typing_anim_job)
        _typing_anim_job = None
    if _typing_frame:
        _typing_frame.destroy()
        _typing_frame = None
    _typing_label = None


def _set_input_state(enabled: bool):
    state  = "normal"  if enabled else "disabled"
    cursor = "hand2"   if enabled else ""
    user_entry.config(state=state)
    send_btn_frame.config(cursor=cursor)
    send_btn_label.config(cursor=cursor,
                          fg=C["send_fg"] if enabled else C["header_sub"])


# ─────────────────────────────────────────
# Send / Receive
# ─────────────────────────────────────────
def send_query(event=None):
    user_query = user_entry.get().strip()
    if not user_query:
        return

    add_message(user_query, "user")
    user_entry.delete(0, tk.END)
    _set_input_state(False)
    _show_typing()

    settings_str = json.dumps({"model": model_var.get()})

    def worker():
        response = chat_openrouter.process_query(user_query, settings_str)
        root.after(0, lambda: _on_response(response))

    def _on_response(backend_response):
        _hide_typing()
        _set_input_state(True)
        user_entry.focus_set()

        try:
            parsed = json.loads(backend_response)
        except Exception:
            parsed = None

        if isinstance(parsed, dict) and ("answer" in parsed or "response" in parsed):
            answer_text = parsed.get("answer") or parsed.get("response") or ""
            add_message(answer_text, "bot",
                        sources_numbered=parsed.get("sources_numbered"))
            add_citations(parsed)
        else:
            add_message(backend_response, "bot")

    threading.Thread(target=worker, daemon=True).start()


def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)

def on_send_hover(_):
    send_btn_frame.config(bg=C["send_hover"])
    send_btn_label.config(bg=C["send_hover"])

def on_send_leave(_):
    send_btn_frame.config(bg=C["send_bg"])
    send_btn_label.config(bg=C["send_bg"])


# ─────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────
print("Initializing main window...")
root = tk.Tk()
root.title("Legal ChatBot")
root.geometry("960x700")
root.minsize(640, 480)
root.configure(bg=C["bg"])

root.rowconfigure(0, weight=0)   # header
root.rowconfigure(1, weight=1)   # chat
root.rowconfigure(2, weight=0)   # controls
root.rowconfigure(3, weight=0)   # separator
root.rowconfigure(4, weight=0)   # input
root.columnconfigure(0, weight=1)

# ── Header ──────────────────────────────
header = tk.Frame(root, bg=C["header_bg"], height=56)
header.grid(row=0, column=0, sticky="ew")
header.grid_propagate(False)
header.columnconfigure(1, weight=1)

tk.Label(header, text="⚖  Legal ChatBot",
         bg=C["header_bg"], fg=C["header_fg"],
         font=FONT_TITLE, padx=18).grid(row=0, column=0, sticky="w")

tk.Label(header, text="Powered by Greek Legal Database",
         bg=C["header_bg"], fg=C["header_sub"],
         font=FONT_SUB, padx=18).grid(row=0, column=2, sticky="e")

# ── Chat Canvas ──────────────────────────
chat_outer = tk.Frame(root, bg=C["canvas_bg"])
chat_outer.grid(row=1, column=0, sticky="nsew")
chat_outer.rowconfigure(0, weight=1)
chat_outer.columnconfigure(0, weight=1)

chat_canvas = tk.Canvas(chat_outer, bg=C["canvas_bg"],
                         borderwidth=0, highlightthickness=0,
                         yscrollincrement=1)   # pixel-level scrolling
vsb = ttk.Scrollbar(chat_outer, orient="vertical", command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=vsb.set)
vsb.grid(row=0, column=1, sticky="ns")
chat_canvas.grid(row=0, column=0, sticky="nsew")

messages_frame = tk.Frame(chat_canvas, bg=C["canvas_bg"])
messages_frame.columnconfigure(0, weight=1)
canvas_window = chat_canvas.create_window((0, 0), window=messages_frame, anchor="nw")

def on_frame_configure(event):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)
messages_frame.bind("<Configure>", on_frame_configure)

def on_canvas_configure(event):
    chat_canvas.itemconfig(canvas_window, width=event.width)
chat_canvas.bind("<Configure>", on_canvas_configure)

_scroll_vel   = 0.0   # pixels/frame
_scroll_job   = None  # pending after() id
_scroll_rem   = 0.0   # sub-pixel remainder

def _animate_scroll():
    global _scroll_vel, _scroll_job, _scroll_rem
    _scroll_rem += _scroll_vel
    px = int(_scroll_rem)
    if px:
        chat_canvas.yview_scroll(px, "units")  # 1 unit = 1 px (yscrollincrement=1)
        _scroll_rem -= px
    _scroll_vel *= 0.88                         # friction — tune 0.80–0.92 to taste
    if abs(_scroll_vel) > 0.4:
        _scroll_job = root.after(16, _animate_scroll)   # ~60 fps
    else:
        _scroll_vel = _scroll_rem = 0.0
        _scroll_job = None

def on_mousewheel(event):
    global _scroll_vel, _scroll_job
    _scroll_vel += -event.delta / 5            # /5 ≈ 24 px per mouse notch
    if _scroll_job is None:
        _animate_scroll()

# bind_all catches scroll events even when hovering over child widgets (Text bubbles, etc.)
root.bind_all("<MouseWheel>", on_mousewheel)

# ── Control Panel ────────────────────────
ctrl = tk.Frame(root, bg=C["ctrl_bg"], pady=7)
ctrl.grid(row=2, column=0, sticky="ew")

tk.Label(ctrl, text="Model:", bg=C["ctrl_bg"], fg=C["header_bg"],
         font=FONT_LABEL).pack(side="left", padx=(16, 6))

model_var     = tk.StringVar(root)
model_options = ["deepseek/deepseek-chat-v3-0324", "other/model:version"]
model_var.set(model_options[0])

# Custom styled dropdown (ttk.OptionMenu ignores colours on macOS Aqua theme)
model_popup = tk.Menu(root, tearoff=0, font=FONT_BODY,
                      bg=C["input_bg"], fg=C["bot_fg"],
                      activebackground=C["user_bg"], activeforeground=C["user_fg"],
                      relief="flat", bd=0)

model_btn_outer = tk.Frame(ctrl, bg=C["border"], padx=1, pady=1)   # 1-px border
model_btn_outer.pack(side="left", padx=(4, 0))

model_btn_inner = tk.Frame(model_btn_outer, bg=C["input_bg"],
                            padx=10, pady=5, cursor="hand2")
model_btn_inner.pack()

model_btn_label = tk.Label(model_btn_inner, textvariable=model_var,
                            font=FONT_BODY, bg=C["input_bg"], fg=C["bot_fg"],
                            cursor="hand2")
model_btn_label.pack(side="left")

tk.Label(model_btn_inner, text=" ▾", font=FONT_SMALL,
         bg=C["input_bg"], fg=C["label_bot"], cursor="hand2").pack(side="left")

def show_model_menu(_=None):
    x = model_btn_outer.winfo_rootx()
    y = model_btn_outer.winfo_rooty() + model_btn_outer.winfo_height() + 2
    model_popup.tk_popup(x, y)

for opt in model_options:
    model_popup.add_command(label=opt, command=lambda v=opt: model_var.set(v))

for w in (model_btn_outer, model_btn_inner, model_btn_label):
    w.bind("<Button-1>", show_model_menu)

# ── Separator ────────────────────────────
tk.Frame(root, bg=C["border"], height=1).grid(row=3, column=0, sticky="ew")

# ── Input Bar ────────────────────────────
input_bar = tk.Frame(root, bg=C["input_bg"], pady=10, padx=12)
input_bar.grid(row=4, column=0, sticky="ew")
input_bar.columnconfigure(0, weight=1)

entry_wrap = tk.Frame(input_bar, bg=C["border"], bd=0)
entry_wrap.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=1, ipadx=1)
entry_wrap.columnconfigure(0, weight=1)

user_entry = tk.Entry(entry_wrap, font=FONT_BODY, bg=C["input_bg"],
                      fg=C["bot_fg"], relief="flat",
                      insertbackground=C["cursor"], bd=7)
user_entry.grid(row=0, column=0, sticky="ew")

send_btn_frame = tk.Frame(input_bar, bg=C["send_bg"], cursor="hand2",
                          padx=22, pady=9)
send_btn_frame.grid(row=0, column=1)
send_btn_label = tk.Label(send_btn_frame, text="Send", font=FONT_SEND,
                           bg=C["send_bg"], fg=C["send_fg"], cursor="hand2")
send_btn_label.pack()
for w in (send_btn_frame, send_btn_label):
    w.bind("<Button-1>", lambda _: send_query())
    w.bind("<Enter>", on_send_hover)
    w.bind("<Leave>", on_send_leave)

user_entry.bind("<Return>", send_query)

# ── Context Menu ─────────────────────────
context_menu = tk.Menu(root, tearoff=0, font=FONT_SMALL)
for cmd, seq in [("Cut", "<<Cut>>"), ("Copy", "<<Copy>>"), ("Paste", "<<Paste>>")]:
    context_menu.add_command(label=cmd, command=lambda s=seq: user_entry.event_generate(s))
user_entry.bind("<Button-3>", show_context_menu)

user_entry.focus_set()
print("Starting main loop...")
root.mainloop()
