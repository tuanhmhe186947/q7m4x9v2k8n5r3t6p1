"use strict";

const state = {
  items: [],
  itemIndex: new Map(),
  selected: new Set(),
  preview: null,
  commitInProgress: false,
  ageSelectionDays: null,
  visibleLimit: 200,
  browseStack: [],
};

const elements = {
  scanButton: document.querySelector("#scan-button"),
  largeReview: document.querySelector("#large-review"),
  search: document.querySelector("#search-input"),
  category: document.querySelector("#category-filter"),
  risk: document.querySelector("#risk-filter"),
  recommendation: document.querySelector("#recommendation-filter"),
  sort: document.querySelector("#sort-filter"),
  ageShortcuts: document.querySelectorAll("[data-age-days]"),
  customAgeDays: document.querySelector("#custom-age-days"),
  applyCustomAge: document.querySelector("#apply-custom-age"),
  selectVisible: document.querySelector("#select-visible"),
  resultBody: document.querySelector("#result-body"),
  emptyState: document.querySelector("#empty-state"),
  notice: document.querySelector("#notice"),
  previewButton: document.querySelector("#preview-button"),
  loadMore: document.querySelector("#load-more"),
  dialog: document.querySelector("#confirm-dialog"),
  previewSummary: document.querySelector("#preview-summary"),
  previewList: document.querySelector("#preview-list"),
  confirmPhrase: document.querySelector("#confirm-phrase"),
  confirmInput: document.querySelector("#confirm-input"),
  commitButton: document.querySelector("#commit-button"),
  dialogClose: document.querySelector("#dialog-close"),
  dialogCancel: document.querySelector("#dialog-cancel"),
  commitProgress: document.querySelector("#commit-progress"),
  commitProgressTitle: document.querySelector("#commit-progress-title"),
  commitProgressCount: document.querySelector("#commit-progress-count"),
  commitProgressBar: document.querySelector("#commit-progress-bar"),
  commitProgressDetail: document.querySelector("#commit-progress-detail"),
  ageScopeDialog: document.querySelector("#age-scope-dialog"),
  ageScopeSummary: document.querySelector("#age-scope-summary"),
  ageScopeList: document.querySelector("#age-scope-list"),
  ageScopeNote: document.querySelector("#age-scope-note"),
  ageScopeClose: document.querySelector("#age-scope-close"),
  ageScopeDismiss: document.querySelector("#age-scope-dismiss"),
  insights: document.querySelector("#category-insights"),
  detailDialog: document.querySelector("#detail-dialog"),
  detailTitle: document.querySelector("#detail-title"),
  detailPath: document.querySelector("#detail-path"),
  detailSummary: document.querySelector("#detail-summary"),
  detailList: document.querySelector("#detail-list"),
  detailNote: document.querySelector("#detail-note"),
  detailBack: document.querySelector("#detail-back"),
  detailClose: document.querySelector("#detail-close"),
};

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(
    Math.floor(Math.log(value) / Math.log(1024)),
    units.length - 1,
  );
  const amount = value / (1024 ** exponent);
  const digits = amount >= 100 || exponent === 0 ? 0 : 1;
  return `${amount.toFixed(digits)} ${units[exponent]}`;
}

function statusLabel(risk) {
  return {
    safe: "Có thể tái tạo",
    caution: "Cần xem kỹ",
    protected: "Được bảo vệ",
  }[risk] || risk;
}

function setNotice(message, isError = false) {
  elements.notice.textContent = message;
  elements.notice.classList.toggle("error", isError);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Phản hồi không hợp lệ (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(data.detail || `Yêu cầu thất bại (${response.status}).`);
  }
  return data;
}

async function getJson(url) {
  const response = await fetch(url);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Phản hồi không hợp lệ (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(data.detail || `Yêu cầu thất bại (${response.status}).`);
  }
  return data;
}

function formatAge(days) {
  if (days < (1 / 24)) {
    return "Vừa cập nhật";
  }
  if (days < 1) {
    return `${Math.max(1, Math.floor(days * 24))} giờ trước`;
  }
  if (days < 60) {
    return `${Math.floor(days)} ngày trước`;
  }
  if (days < 730) {
    return `${Math.floor(days / 30)} tháng trước`;
  }
  return `${Math.floor(days / 365)} năm trước`;
}

function formatDate(value) {
  return new Date(value).toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function filteredItems() {
  const query = elements.search.value.trim().toLocaleLowerCase("vi");
  const items = state.items.filter((item) => {
    const matchesText = !query
      || item.display_name.toLocaleLowerCase("vi").includes(query)
      || item.path.toLocaleLowerCase("vi").includes(query);
    const matchesCategory = !elements.category.value
      || item.category === elements.category.value;
    const matchesRisk = !elements.risk.value
      || item.risk === elements.risk.value;
    const matchesRecommendation = !elements.recommendation.value
      || item.recommendation_level === elements.recommendation.value;
    return matchesText
      && matchesCategory
      && matchesRisk
      && matchesRecommendation;
  });
  const comparators = {
    age_desc: (left, right) => right.age_days - left.age_days,
    size_desc: (left, right) => right.size_bytes - left.size_bytes,
    modified_desc: (left, right) => left.age_days - right.age_days,
    name_asc: (left, right) => {
      return left.display_name.localeCompare(right.display_name, "vi");
    },
  };
  return items.sort(comparators[elements.sort.value] || comparators.age_desc);
}

function visibleItems() {
  return filteredItems().slice(0, state.visibleLimit);
}

function createCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function pathContains(parent, child) {
  const normalizedParent = parent.replaceAll("/", "\\").toLocaleLowerCase();
  const normalizedChild = child.replaceAll("/", "\\").toLocaleLowerCase();
  const prefix = normalizedParent.endsWith("\\")
    ? normalizedParent
    : `${normalizedParent}\\`;
  return normalizedChild === normalizedParent
    || normalizedChild.startsWith(prefix);
}

function setItemSelected(item, checked) {
  if (!checked) {
    state.selected.delete(item.token);
    return;
  }
  for (const token of Array.from(state.selected)) {
    const other = state.itemIndex.get(token);
    if (!other) {
      continue;
    }
    if (pathContains(item.path, other.path) || pathContains(other.path, item.path)) {
      state.selected.delete(token);
    }
  }
  state.selected.add(item.token);
}

function syncSelectionControls() {
  document.querySelectorAll("input[data-token]").forEach((checkbox) => {
    checkbox.checked = state.selected.has(checkbox.dataset.token);
  });
}

function createAgeCell(item) {
  const cell = document.createElement("td");
  const wrap = document.createElement("div");
  wrap.className = "age-cell";
  const relative = document.createElement("strong");
  relative.textContent = formatAge(item.age_days);
  const exact = document.createElement("span");
  exact.textContent = formatDate(item.modified_at);
  wrap.append(relative, exact);
  cell.append(wrap);
  return cell;
}

function createRecommendation(item, compact = false) {
  const wrap = document.createElement("div");
  wrap.className = [
    "recommendation",
    `recommendation-${item.recommendation_level}`,
    compact ? "recommendation-compact" : "",
  ].filter(Boolean).join(" ");
  const label = document.createElement("span");
  label.className = "recommendation-label";
  label.textContent = item.recommendation;
  const importance = document.createElement("span");
  importance.className = `importance importance-${item.importance_level}`;
  importance.textContent = `Dự án: ${importanceLabel(item.importance_level)}`;
  importance.title = item.importance_reason;
  const impact = document.createElement("span");
  impact.className = "recommendation-impact";
  impact.textContent = item.importance_reason;
  wrap.append(label, importance, impact);
  return wrap;
}

function importanceLabel(level) {
  return {
    critical: "rất quan trọng",
    high: "quan trọng",
    medium: "cần xem",
    low: "thấp",
  }[level] || "chưa rõ";
}

function renderRows() {
  const filtered = filteredItems();
  const visible = filtered.slice(0, state.visibleLimit);
  elements.resultBody.replaceChildren();
  for (const item of visible) {
    const row = document.createElement("tr");
    const checkCell = document.createElement("td");
    checkCell.className = "check-column";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(item.token);
    checkbox.disabled = !item.selectable;
    checkbox.dataset.token = item.token;
    checkbox.setAttribute("aria-label", `Chọn ${item.display_name}`);
    checkbox.addEventListener("change", () => {
      setItemSelected(item, checkbox.checked);
      syncSelectionControls();
      updateSelectionSummary();
    });
    checkCell.append(checkbox);
    row.append(checkCell);

    const nameCell = document.createElement("td");
    const nameWrap = document.createElement("div");
    nameWrap.className = "item-name";
    const name = document.createElement("strong");
    name.textContent = item.display_name;
    name.title = item.reason;
    const path = document.createElement("span");
    path.textContent = item.path;
    path.title = item.protected_reason || item.reason;
    nameWrap.append(name, path, createRecommendation(item));
    nameCell.append(nameWrap);
    row.append(nameCell);

    row.append(createCell(item.category_label));
    row.append(createCell(formatBytes(item.size_bytes)));
    row.append(createAgeCell(item));

    const riskCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `risk-badge risk-${item.risk}`;
    badge.textContent = statusLabel(item.risk);
    badge.title = item.protected_reason || item.reason;
    riskCell.append(badge);
    row.append(riskCell);

    const exploreCell = document.createElement("td");
    if (item.can_browse) {
      const explore = document.createElement("button");
      explore.className = "explore-button";
      explore.type = "button";
      explore.textContent = "Mở sâu";
      explore.addEventListener("click", () => openDetails(item));
      exploreCell.append(explore);
    }
    row.append(exploreCell);
    elements.resultBody.append(row);
  }
  elements.emptyState.classList.toggle("hidden", visible.length > 0);
  const remaining = filtered.length - visible.length;
  elements.loadMore.hidden = remaining <= 0;
  elements.loadMore.textContent = remaining > 0
    ? `Hiện thêm ${Math.min(200, remaining)} (${remaining} mục còn lại)`
    : "Hiện thêm";
  updateSelectVisibleControl();
}

function updateSelectVisibleControl() {
  const selectable = visibleItems().filter((item) => item.selectable);
  const allSelected = selectable.length > 0
    && selectable.every((item) => state.selected.has(item.token));
  const action = allSelected ? "Bỏ chọn" : "Chọn";
  elements.selectVisible.disabled = selectable.length === 0;
  elements.selectVisible.textContent = `${action} ${selectable.length} mục`;
  elements.selectVisible.title = `${action} các mục đủ điều kiện đang hiển thị`;
  elements.selectVisible.setAttribute(
    "aria-label",
    `${action} ${selectable.length} mục đủ điều kiện đang hiển thị`,
  );
}

function updateSelectionSummary() {
  const selectedItems = Array.from(state.selected)
    .map((token) => state.itemIndex.get(token))
    .filter(Boolean);
  const selectedBytes = selectedItems.reduce((sum, item) => {
    return sum + item.size_bytes;
  }, 0);
  document.querySelector("#selected-size").textContent = formatBytes(selectedBytes);
  document.querySelector("#selected-count").textContent =
    `${selectedItems.length} mục`;
  document.querySelector("#action-summary").textContent = selectedItems.length
    ? `${selectedItems.length} mục · ${formatBytes(selectedBytes)}`
    : "Chưa chọn mục nào";
  elements.previewButton.disabled = selectedItems.length === 0;
  updateSelectVisibleControl();
}

function updateCategoryOptions() {
  const existing = new Set(
    Array.from(elements.category.options).map((option) => option.value),
  );
  const categories = new Map();
  for (const item of state.items) {
    categories.set(item.category, item.category_label);
  }
  for (const [value, label] of categories) {
    if (existing.has(value)) {
      continue;
    }
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    elements.category.append(option);
  }
}

function renderInsights() {
  const groups = new Map();
  for (const item of state.items) {
    const current = groups.get(item.category) || {
      category: item.category,
      label: item.category_label,
      bytes: 0,
      selectableBytes: 0,
      count: 0,
      selectableCount: 0,
    };
    current.bytes += item.size_bytes;
    current.count += 1;
    if (item.selectable) {
      current.selectableBytes += item.size_bytes;
      current.selectableCount += 1;
    }
    groups.set(item.category, current);
  }
  const priority = ["agent_worktrees", "codex_sessions"];
  const ordered = [
    ...priority.map((key) => groups.get(key)).filter(Boolean),
    ...Array.from(groups.values())
      .filter((group) => !priority.includes(group.category))
      .sort((left, right) => right.bytes - left.bytes),
  ].slice(0, 4);
  elements.insights.replaceChildren();
  for (const group of ordered) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "insight-card";
    const label = document.createElement("span");
    label.textContent = `${group.label} · ${group.count} mục`;
    const size = document.createElement("strong");
    size.textContent = formatBytes(group.bytes);
    const action = document.createElement("small");
    action.textContent = group.selectableCount
      ? `${formatBytes(group.selectableBytes)} có thể chọn →`
      : "Chỉ xem hoặc đang được bảo vệ →";
    card.append(label, size, action);
    card.addEventListener("click", () => {
      elements.category.value = group.category;
      state.visibleLimit = 200;
      renderRows();
      document.querySelector(".workspace").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
    elements.insights.append(card);
  }
}

function updateScanSummary(payload) {
  const summary = payload.summary;
  const usedPercent = summary.disk_total_bytes
    ? (summary.disk_used_bytes / summary.disk_total_bytes) * 100
    : 0;
  document.querySelector("#disk-used").textContent =
    `${usedPercent.toFixed(1)}%`;
  document.querySelector("#disk-meter").style.width =
    `${Math.min(usedPercent, 100)}%`;
  document.querySelector("#disk-detail").textContent =
    `${formatBytes(summary.disk_free_bytes)} còn trống`;
  document.querySelector("#available-size").textContent =
    formatBytes(summary.delete_first_bytes);
  document.querySelector("#available-count").textContent =
    `${summary.delete_first_count} mục cache/tạm`;
  document.querySelector("#review-size").textContent =
    formatBytes(summary.project_critical_bytes);
  document.querySelector("#review-count").textContent =
    `${summary.project_critical_count} mục cần bảo vệ`;
  document.querySelector("#protected-size").textContent =
    formatBytes(summary.protected_bytes);
  document.querySelector("#protected-count").textContent =
    `${summary.protected_count} mục active/khoa học`;
  const scannedAt = new Date(payload.scanned_at);
  document.querySelector("#scan-time").textContent =
    `Quét lúc ${scannedAt.toLocaleString("vi-VN")}`;
}

async function runScan() {
  elements.scanButton.disabled = true;
  elements.scanButton.textContent = "Đang quét…";
  setNotice("Đang đo các vị trí đã đăng ký. Quá trình này chỉ đọc filesystem.");
  try {
    const payload = await postJson("/api/scan", {
      include_large_review: elements.largeReview.checked,
    });
    state.items = payload.items;
    updateAgeSelectionControls();
    state.itemIndex = new Map(
      payload.items.map((item) => [item.token, item]),
    );
    state.selected.clear();
    state.ageSelectionDays = null;
    setAgeSelectionIndicator(null);
    state.preview = null;
    state.visibleLimit = 200;
    updateCategoryOptions();
    renderInsights();
    updateScanSummary(payload);
    renderRows();
    updateSelectionSummary();
    const errorNote = payload.errors.length
      ? ` Có ${payload.errors.length} vị trí không đọc được và đã được bỏ qua.`
      : "";
    setNotice(
      `Đã phát hiện ${payload.summary.item_count} mục.${errorNote}`
      + " Không có mục nào được tự động chọn.",
    );
  } catch (error) {
    setNotice(error.message, true);
  } finally {
    elements.scanButton.disabled = false;
    elements.scanButton.replaceChildren();
    const icon = document.createElement("span");
    icon.className = "button-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "↻";
    elements.scanButton.append(icon, " Quét lại");
  }
}

function selectVisible() {
  const visible = visibleItems().filter((item) => item.selectable);
  const allSelected = visible.length > 0
    && visible.every((item) => state.selected.has(item.token));
  for (const item of visible) {
    if (allSelected) {
      setItemSelected(item, false);
    } else {
      setItemSelected(item, true);
    }
  }
  syncSelectionControls();
  updateSelectionSummary();
}

function selectSafeByAge(minAgeDays) {
  if (!state.items.length) {
    setNotice("Hãy quét ổ đĩa trước khi chọn theo số ngày.", true);
    return false;
  }
  const matchingItems = filteredItems().filter((item) => item.age_days >= minAgeDays);
  const safeItems = matchingItems
    .filter((item) => (
      item.selectable
      && item.recommendation_level === "delete_first"
      && item.importance_level === "low"
      && item.age_days >= minAgeDays
    ))
    .sort((left, right) => right.age_days - left.age_days);
  const candidates = safeItems.slice(0, 500);
  const safeTokens = new Set(safeItems.map((item) => item.token));
  const skippedItems = matchingItems
    .filter((item) => !safeTokens.has(item.token))
    .sort((left, right) => right.size_bytes - left.size_bytes);
  const truncatedCount = safeItems.length - candidates.length;
  state.selected.clear();
  for (const item of candidates) {
    setItemSelected(item, true);
  }
  syncSelectionControls();
  updateSelectionSummary();
  renderAgeSelectionScope(
    minAgeDays,
    matchingItems,
    candidates,
    skippedItems,
    truncatedCount,
  );
  const scopeLabel = matchingItems.length
    ? `${candidates.length}/${matchingItems.length} mục khớp bộ lọc hiện tại`
    : "Không có mục nào khớp bộ lọc hiện tại";
  setNotice(
    `${scopeLabel} từ ${minAgeDays} ngày trước đã được áp dụng.`
    + (skippedItems.length ? ` Bỏ qua ${skippedItems.length} mục cần xem.` : ""),
  );
  return candidates.length > 0;
}

function renderAgeSelectionScope(
  minAgeDays,
  matchingItems,
  candidates,
  skippedItems,
  truncatedCount,
) {
  elements.ageScopeSummary.textContent =
    `${candidates.length}/${matchingItems.length} mục đủ tuổi đã được chọn an toàn `
    + `(từ ${minAgeDays} ngày trước).`;
  elements.ageScopeList.replaceChildren();
  const previewItems = skippedItems.slice(0, 50);
  for (const item of previewItems) {
    const row = document.createElement("div");
    row.className = "preview-item";
    const name = document.createElement("span");
    name.textContent = item.display_name;
    name.title = item.path;
    const reason = document.createElement("strong");
    reason.textContent = item.protected_reason
      || item.importance_reason
      || item.recommendation;
    row.append(name, reason);
    elements.ageScopeList.append(row);
  }
  if (!previewItems.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Không có mục quan trọng hoặc bị bảo vệ trong phạm vi này.";
    elements.ageScopeList.append(empty);
  }
  const notes = [];
  if (skippedItems.length) {
    notes.push(
      `${skippedItems.length} mục bị bỏ qua vì là dữ liệu dự án, session, `
      + "worktree hoặc chưa được đánh giá là an toàn.",
    );
  }
  if (truncatedCount > 0) {
    notes.push(
      `${truncatedCount} mục an toàn còn lại chưa chọn trong lô này; `
      + "hãy xử lý lô hiện tại trước khi chọn tiếp.",
    );
  }
  elements.ageScopeNote.textContent = notes.join(" ")
    || "Các mục đã chọn vẫn cần xem trước và xác nhận bằng cụm từ bắt buộc.";
  if (!elements.ageScopeDialog.open) {
    elements.ageScopeDialog.showModal();
  }
}

function setAgeSelectionIndicator(days) {
  state.ageSelectionDays = days;
  elements.ageShortcuts.forEach((button) => {
    const active = days !== null && Number(button.dataset.ageDays) === days;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function updateAgeSelectionControls() {
  const disabled = state.items.length === 0;
  elements.ageShortcuts.forEach((button) => {
    button.disabled = disabled;
  });
  elements.customAgeDays.disabled = disabled;
  elements.applyCustomAge.disabled = disabled;
}

function requestSafeAgeSelection(value) {
  const minAgeDays = Number(value);
  if (!Number.isSafeInteger(minAgeDays) || minAgeDays < 1 || minAgeDays > 36500) {
    setNotice("Nhập số ngày nguyên từ 1 đến 36.500.", true);
    elements.customAgeDays.focus();
    return;
  }
  const applied = selectSafeByAge(minAgeDays);
  setAgeSelectionIndicator(applied ? minAgeDays : null);
}

function renderPreview(preview) {
  elements.previewSummary.textContent =
    `${preview.item_count} mục (${formatBytes(preview.total_bytes)}) sẽ được `
    + "chuyển vào Windows Recycle Bin. Có thể khôi phục từ Recycle Bin.";
  elements.previewList.replaceChildren();
  for (const item of preview.items) {
    const row = document.createElement("div");
    row.className = "preview-item";
    const name = document.createElement("span");
    name.textContent = item.display_name;
    name.title = item.path;
    const size = document.createElement("strong");
    size.textContent = formatBytes(item.size_bytes);
    row.append(name, size);
    elements.previewList.append(row);
  }
  elements.confirmPhrase.textContent = preview.phrase;
  elements.confirmInput.value = "";
  elements.commitButton.disabled = true;
}

function createDetailAge(item) {
  const wrap = document.createElement("div");
  wrap.className = "age-cell";
  const relative = document.createElement("strong");
  relative.textContent = formatAge(item.age_days);
  const exact = document.createElement("span");
  exact.textContent = formatDate(item.modified_at);
  wrap.append(relative, exact);
  return wrap;
}

function renderDetail(payload) {
  const parent = payload.parent;
  elements.detailTitle.textContent = parent.display_name;
  elements.detailPath.textContent = parent.path;
  elements.detailSummary.textContent =
    `${payload.total_count} mục · ${formatBytes(parent.size_bytes)}`;
  elements.detailBack.disabled = state.browseStack.length <= 1;
  elements.detailList.replaceChildren();
  for (const item of payload.items) {
    state.itemIndex.set(item.token, item);
    const row = document.createElement("div");
    row.className = "detail-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(item.token);
    checkbox.disabled = !item.selectable;
    checkbox.dataset.token = item.token;
    checkbox.setAttribute("aria-label", `Chọn ${item.display_name}`);
    checkbox.addEventListener("change", () => {
      setItemSelected(item, checkbox.checked);
      syncSelectionControls();
      updateSelectionSummary();
    });

    const nameWrap = document.createElement("div");
    nameWrap.className = "detail-name";
    const name = document.createElement("strong");
    name.textContent = item.display_name;
    name.title = item.reason;
    const kind = document.createElement("span");
    kind.textContent = item.kind === "directory" ? "Thư mục" : "Tệp";
    nameWrap.append(name, kind, createRecommendation(item, true));

    const size = document.createElement("span");
    size.textContent = formatBytes(item.size_bytes);
    const age = createDetailAge(item);
    const badge = document.createElement("span");
    badge.className = `risk-badge risk-${item.risk}`;
    badge.textContent = statusLabel(item.risk);
    badge.title = item.protected_reason || item.reason;
    const open = document.createElement("span");
    if (item.can_browse) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "explore-button";
      button.textContent = "Mở";
      button.addEventListener("click", () => loadDetail(item, true));
      open.append(button);
    }
    row.append(checkbox, nameWrap, size, age, badge, open);
    elements.detailList.append(row);
  }
  const notes = [];
  if (parent.detail) {
    notes.push(parent.detail);
  }
  if (parent.protected_reason) {
    notes.push(parent.protected_reason);
  } else {
    notes.push(parent.reason);
  }
  if (payload.truncated) {
    notes.push("Chỉ hiển thị 500 mục lớn nhất ở tầng này.");
  }
  elements.detailNote.textContent = notes.join(" · ");
}

async function loadDetail(item, pushHistory) {
  elements.detailList.replaceChildren();
  elements.detailSummary.textContent = "Đang đo bên trong…";
  elements.detailNote.textContent = "";
  try {
    const payload = await getJson(
      `/api/items/${encodeURIComponent(item.token)}/children`,
    );
    if (pushHistory) {
      state.browseStack.push(payload.parent);
    }
    renderDetail(payload);
  } catch (error) {
    elements.detailSummary.textContent = "Không thể mở thư mục";
    elements.detailNote.textContent = error.message;
  }
}

async function openDetails(item) {
  state.browseStack = [];
  elements.detailDialog.showModal();
  await loadDetail(item, true);
}

async function openPreview() {
  elements.previewButton.disabled = true;
  try {
    const preview = await postJson("/api/recycle/preview", {
      tokens: Array.from(state.selected),
    });
    state.preview = preview;
    renderPreview(preview);
    elements.commitProgress.hidden = true;
    elements.dialog.showModal();
    elements.confirmInput.focus();
  } catch (error) {
    setNotice(error.message, true);
  } finally {
    elements.previewButton.disabled = state.selected.size === 0;
  }
}

function renderCommitProgress(job) {
  const total = Math.max(Number(job.total_count) || 0, 1);
  const completed = Math.min(Number(job.completed_count) || 0, total);
  elements.commitProgress.hidden = false;
  elements.commitProgressBar.max = total;
  elements.commitProgressBar.value = completed;
  elements.commitProgressCount.textContent = `${completed}/${job.total_count || 0}`;
  if (job.status === "complete") {
    elements.commitProgressTitle.textContent = "Đã chuyển xong";
    elements.commitProgressDetail.textContent = (
      `${job.recycled_count} mục vào Recycle Bin · `
      + `${formatBytes(job.reclaimed_bytes)} được giải phóng`
    );
    return;
  }
  elements.commitProgressTitle.textContent = "Đang chuyển vào Recycle Bin";
  elements.commitProgressDetail.textContent = job.current_name
    ? `Đang xử lý: ${job.current_name}`
    : "Đang xác nhận an toàn của các mục đã chọn…";
}

async function waitForCommit(jobId) {
  while (true) {
    const job = await getJson(
      `/api/recycle/jobs/${encodeURIComponent(jobId)}`,
    );
    renderCommitProgress(job);
    if (job.status === "complete") {
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Không thể chuyển các mục đã chọn.");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 250));
  }
}

async function commitRecycle() {
  if (!state.preview) {
    return;
  }
  state.commitInProgress = true;
  elements.commitButton.disabled = true;
  elements.dialogClose.disabled = true;
  elements.dialogCancel.disabled = true;
  elements.commitButton.textContent = "Đang chuyển…";
  elements.commitProgress.hidden = false;
  elements.commitProgressBar.removeAttribute("value");
  elements.commitProgressTitle.textContent = "Đang chuẩn bị chuyển";
  elements.commitProgressCount.textContent = "…";
  elements.commitProgressDetail.textContent = "Đang tạo tiến trình an toàn…";
  try {
    const started = await postJson("/api/recycle/commit", {
      confirmation_id: state.preview.confirmation_id,
      phrase: elements.confirmInput.value,
    });
    const result = await waitForCommit(started.job_id);
    state.preview = null;
    state.selected.clear();
    setNotice(
      `Đã chuyển ${result.recycled_count} mục `
      + `(${formatBytes(result.reclaimed_bytes)}) vào Recycle Bin.`
      + (result.failed_count ? ` ${result.failed_count} mục thất bại.` : ""),
      result.failed_count > 0,
    );
    await new Promise((resolve) => window.setTimeout(resolve, 1800));
    elements.dialog.close();
    elements.commitProgress.hidden = true;
    await runScan();
  } catch (error) {
    setNotice(error.message, true);
    elements.dialog.close();
  } finally {
    state.commitInProgress = false;
    elements.commitButton.textContent = "Chuyển các mục đã chọn";
    elements.dialogClose.disabled = false;
    elements.dialogCancel.disabled = false;
  }
}

elements.scanButton.addEventListener("click", runScan);
elements.selectVisible.addEventListener("click", selectVisible);
elements.ageShortcuts.forEach((button) => {
  button.addEventListener("click", () => {
    requestSafeAgeSelection(button.dataset.ageDays);
  });
});
elements.applyCustomAge.addEventListener("click", () => {
  requestSafeAgeSelection(elements.customAgeDays.value);
});
elements.customAgeDays.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    requestSafeAgeSelection(elements.customAgeDays.value);
  }
});
updateAgeSelectionControls();
elements.previewButton.addEventListener("click", openPreview);
elements.commitButton.addEventListener("click", commitRecycle);
elements.ageScopeClose.addEventListener("click", () => {
  elements.ageScopeDialog.close();
});
elements.ageScopeDismiss.addEventListener("click", () => {
  elements.ageScopeDialog.close();
});
elements.detailClose.addEventListener("click", () => {
  elements.detailDialog.close();
});
elements.detailBack.addEventListener("click", async () => {
  if (state.browseStack.length <= 1) {
    return;
  }
  state.browseStack.pop();
  const parent = state.browseStack[state.browseStack.length - 1];
  await loadDetail(parent, false);
});
elements.loadMore.addEventListener("click", () => {
  state.visibleLimit += 200;
  renderRows();
});
elements.confirmInput.addEventListener("input", () => {
  elements.commitButton.disabled = !state.preview
    || elements.confirmInput.value !== state.preview.phrase;
});
elements.dialog.addEventListener("cancel", (event) => {
  if (state.commitInProgress) {
    event.preventDefault();
  }
});

for (const control of [
  elements.search,
  elements.category,
  elements.risk,
  elements.recommendation,
  elements.sort,
]) {
  control.addEventListener("input", () => {
    state.visibleLimit = 200;
    renderRows();
  });
  control.addEventListener("change", () => {
    state.visibleLimit = 200;
    renderRows();
  });
}

// Override the age picker with scoped, project-aware selection semantics.
function ageSkipReason(item) {
  if (!item.selectable) {
    return item.protected_reason || "Được bảo vệ";
  }
  if (item.importance_level !== "low") {
    return item.importance_reason || "Có thể quan trọng với dự án";
  }
  if (item.recommendation_level !== "delete_first") {
    return item.recommendation || "Không nằm trong nhóm dọn trước";
  }
  return "Không đủ điều kiện chọn tự động";
}

function selectSafeByAge(minAgeDays) {
  if (!state.items.length) {
    setNotice("Hãy quét ổ đĩa trước khi chọn theo số ngày.", true);
    return false;
  }

  const matchingItems = filteredItems().filter(
    (item) => item.age_days >= minAgeDays,
  );
  const safeItems = matchingItems
    .filter(
      (item) =>
        item.selectable &&
        item.recommendation_level === "delete_first" &&
        item.importance_level === "low",
    )
    .sort((left, right) => right.age_days - left.age_days);
  const candidates = safeItems.slice(0, 500);
  const selectedTokens = new Set(candidates.map((item) => item.token));
  const safeTokens = new Set(safeItems.map((item) => item.token));
  const skippedItems = matchingItems
    .filter((item) => !safeTokens.has(item.token))
    .sort((left, right) => right.size_bytes - left.size_bytes);
  const delayedSafeCount = safeItems.length - candidates.length;

  state.selected.clear();
  for (const item of candidates) {
    setItemSelected(item, true);
  }

  syncSelectionControls();
  updateSelectionSummary();
  renderAgeSelectionScope(
    minAgeDays,
    matchingItems,
    candidates,
    skippedItems,
    delayedSafeCount,
  );

  const selectedCount = selectedTokens.size;
  const scopeLabel = matchingItems.length
    ? `${selectedCount}/${matchingItems.length} mục trong bộ lọc hiện tại`
    : "Không có mục nào trong bộ lọc hiện tại";
  const skipLabel =
    skippedItems.length || delayedSafeCount
      ? `; bỏ qua ${skippedItems.length + delayedSafeCount} mục`
      : "";
  setNotice(`Đã áp dụng ≥ ${minAgeDays} ngày: ${scopeLabel}${skipLabel}.`);
  return candidates.length > 0;
}

function renderAgeSelectionScope(
  minAgeDays,
  matchingItems,
  candidates,
  skippedItems,
  delayedSafeCount,
) {
  elements.ageScopeSummary.textContent =
    `Đang xét ${matchingItems.length} mục thỏa bộ lọc hiện tại và tuổi ` +
    `≥ ${minAgeDays} ngày. Đã chọn ${candidates.length} mục an toàn ` +
    "trong nhóm dọn trước; mục quan trọng với dự án không được chọn tự động.";
  elements.ageScopeList.replaceChildren();

  const previewItems = skippedItems.slice(0, 50);
  for (const item of previewItems) {
    const row = document.createElement("div");
    row.className = "preview-item";
    const name = document.createElement("span");
    name.textContent = item.display_name;
    name.title = item.path;
    const reason = document.createElement("strong");
    reason.textContent = ageSkipReason(item);
    row.append(name, reason);
    elements.ageScopeList.append(row);
  }

  if (!previewItems.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Không có mục quan trọng nào bị bỏ qua trong phạm vi này.";
    elements.ageScopeList.append(empty);
  }

  const notes = [];
  if (skippedItems.length) {
    notes.push(
      `${skippedItems.length} mục khớp tuổi nhưng không được chọn vì đang ` +
        "được bảo vệ, cần xem, hoặc liên quan lineage/worktree/session.",
    );
  }
  if (delayedSafeCount > 0) {
    notes.push(
      `${delayedSafeCount} mục an toàn còn lại chưa chọn trong lô này vì ` +
        "mỗi lần xác nhận giới hạn 500 mục. Hãy xử lý lô hiện tại rồi chọn tiếp.",
    );
  }
  elements.ageScopeNote.textContent =
    notes.join(" ") ||
    "Các mục đã chọn vẫn cần xem lại và xác nhận bằng cụm từ bắt buộc.";

  if (!elements.ageScopeDialog.open) {
    elements.ageScopeDialog.showModal();
  }
}
