const response = await fetch('./demos.json');
if (!response.ok) throw new Error(`Gallery HTTP ${response.status}`);
const demos = await response.json();
const grid = document.getElementById('demos');
for (const demo of demos) {
  const card = document.createElement('a');
  card.className = 'card';
  card.href = `./demos/${demo.id}/`;
  const img = document.createElement('img');
  img.src = `./previews/${demo.id}.gif`;
  img.alt = demo.alt;
  img.loading = 'lazy';
  img.width = 640; img.height = 360;
  const body = document.createElement('div'); body.className = 'card-body';
  const title = document.createElement('div'); title.className = 'card-title';
  const heading = document.createElement('h2'); heading.textContent = demo.title;
  const enter = document.createElement('span'); enter.className = 'enter'; enter.textContent = 'Enter room →';
  title.append(heading, enter);
  const description = document.createElement('p'); description.textContent = demo.description;
  const metrics = document.createElement('div'); metrics.className = 'metrics';
  for (const text of [`${demo.objects} movable object${demo.objects === 1 ? '' : 's'}`, `${demo.download_mb.toFixed(1)} MB room`, 'Eyeful Tower · MIT']) {
    const span = document.createElement('span'); span.textContent = text; metrics.append(span);
  }
  body.append(title, description, metrics); card.append(img, body); grid.append(card);
}
