async (page) => {
  const origin = 'http://127.0.0.1:7867';
  const config = await (await page.request.get(origin + '/api/config')).json();
  if (!config.mock) throw new Error('Mock preview required; never run models locally');
  async function post(path, body = {}) {
    const response = await page.request.post(origin + '/api' + path, {data: body, headers: {'Idempotency-Key': crypto.randomUUID()}});
    if (!response.ok()) throw new Error(await response.text());
    return response.json();
  }
  async function done(id) {
    for (let i = 0; i < 500; i++) {
      const row = await (await page.request.get(origin + '/api/jobs/' + id)).json();
      if (!['queued', 'running'].includes(row.status)) return row;
      await page.waitForTimeout(100);
    }
    throw new Error('Mock job timed out');
  }
  const avatar = await post('/avatars', {name: 'Batch management QA', mode: 'Custom'});
  for (const kind of ['face', 'body']) {
    const candidate = await done((await post('/jobs', {kind: 'reference', avatar_id: avatar.id, reference_kind: kind})).id);
    if (candidate.status !== 'completed') throw new Error(JSON.stringify(candidate));
    const locked = await done((await post('/jobs', {kind: 'confirm', avatar_id: avatar.id, reference_kind: kind, image: candidate.result.candidate})).id);
    if (locked.status !== 'completed') throw new Error(JSON.stringify(locked));
  }
  await page.setViewportSize({width:1366, height:900});
  await page.goto(origin + '/avatars/' + avatar.id + '/gallery/generations');
  async function create(name, quantity, stop) {
    await page.getByRole('button', {name: 'Create batch', exact: true}).click();
    const dialog = page.getByRole('dialog', {name:'Gallery batch', exact:true});
    await dialog.getByRole('textbox', {name:'Batch name', exact:true}).fill(name);
    await dialog.getByRole('textbox', {name:'Theme', exact:true}).fill('Independent cafe theme');
    const slider = dialog.getByRole('slider', {name:'Photographs', exact:true});
    await slider.press('Home');
    for (let i=1; i<quantity; i++) await slider.press('ArrowRight');
    const response = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST');
    await dialog.getByRole('button', {name:'Start batch',exact:true}).click();
    const job = await (await response).json();
    if (!job.payload.new_batch || !job.payload.batch_id) throw new Error('Fresh batch was not explicitly reserved');
    if (stop) await post('/jobs/' + job.id + '/cancel');
    const result = await done(job.id);
    if (!stop && result.status !== 'completed') throw new Error(JSON.stringify(result));
    await page.getByRole('dialog', {name:'Reference pipeline', exact:true}).getByRole('button', {name:'Close', exact:true}).click();
    return job;
  }
  const a = await create('Paused A', 4, true);
  const b = await create('Independent B', 1, false);
  await page.getByRole('button', {name:'Batches',exact:true}).click();
  let manager = page.getByRole('dialog', {name:'Gallery batches',exact:true});
  await manager.getByRole('heading', {name:'Paused A',exact:true}).waitFor();
  await manager.getByRole('heading', {name:'Independent B',exact:true}).waitFor();
  let aRow = manager.getByRole('region', {name:'Batch Paused A',exact:true});
  await aRow.getByRole('button', {name:'Resume batch',exact:true}).click();
  await page.getByRole('dialog', {name:'Reference pipeline',exact:true}).waitFor();
  for (let i=0;i<500;i++) {
    const jobs = await (await page.request.get(origin + '/api/jobs')).json();
    const resumed = jobs.find(j => j.payload.batch_id === a.payload.batch_id && j.id !== a.id);
    if (resumed) { const result = await done(resumed.id); if(result.status !== 'completed') throw new Error(JSON.stringify(result)); break; }
    if(i === 499) throw new Error('Resume did not submit');
    await page.waitForTimeout(100);
  }
  await page.getByRole('dialog', {name:'Reference pipeline',exact:true}).getByRole('button', {name:'Close',exact:true}).click();
  const aPage = await (await page.request.get(origin + '/api/avatars/' + avatar.id + '/gallery?batch_id=' + a.payload.batch_id)).json();
  const bPage = await (await page.request.get(origin + '/api/avatars/' + avatar.id + '/gallery?batch_id=' + b.payload.batch_id)).json();
  if(aPage.total !== 4 || bPage.total !== 1) throw new Error('Batches mixed or resume duplicated outputs');
  await page.getByRole('button', {name:'Batches',exact:true}).click();
  manager = page.getByRole('dialog', {name:'Gallery batches',exact:true});
  aRow = manager.getByRole('region', {name:'Batch Paused A',exact:true});
  page.once('dialog', dialog => dialog.accept());
  await aRow.getByRole('button', {name:'Archive batch',exact:true}).click();
  await manager.getByRole('combobox', {name:'Batch status',exact:true}).selectOption('archived');
  await manager.getByRole('region', {name:'Batch Paused A',exact:true}).getByRole('button', {name:'Restore batch',exact:true}).click();
  await manager.getByRole('combobox', {name:'Batch status',exact:true}).selectOption('active');
  const bRow = manager.getByRole('region', {name:'Batch Independent B',exact:true});
  await bRow.getByRole('button', {name:'Rename batch',exact:true}).click();
  await bRow.getByRole('textbox', {name:'Batch name',exact:true}).fill('Renamed B');
  await bRow.getByRole('button', {name:'Save batch name',exact:true}).click();
  await manager.getByRole('heading', {name:'Renamed B',exact:true}).waitFor();
  for (const width of [1366,390,320]) {
    await page.setViewportSize({width,height:900});
    if(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Batch manager overflows viewport');
    await page.screenshot({path:`output/playwright/batch-manager-${width}.png`});
  }
  await manager.getByRole('region', {name:'Batch Renamed B',exact:true}).getByRole('button', {name:'Open batch',exact:true}).click();
  await page.waitForURL('**?batch=' + b.payload.batch_id);
  await page.reload();
  if(await page.getByRole('combobox', {name:'Gallery batch',exact:true}).inputValue() !== b.payload.batch_id) throw new Error('Batch selection lost on reload');
  await page.getByRole('tab', {name:/Resources/}).click();
  if(new URL(page.url()).searchParams.get('batch') !== b.payload.batch_id) throw new Error('Resource tab lost batch scope');
  await page.getByRole('tab', {name:'Identity',exact:true}).click();
  await page.getByRole('tab', {name:'Gallery',exact:true}).click();
  await page.getByRole('combobox', {name:'Gallery batch',exact:true}).locator('option', {hasText:'Renamed B'}).waitFor({state:'attached'});
  await page.setViewportSize({width:1366,height:900});
  return {passed:true, avatar_id:avatar.id, checks:['fresh batches','paused A does not block B','explicit resume','isolated resources and outputs','archive/restore','rename','scoped reload','desktop/mobile']};
}
