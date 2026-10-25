async (page) => {
  const origin = 'http://127.0.0.1:7867';
  const config = await (await page.request.get(origin + '/api/config')).json();
  if (!config.mock) throw new Error('This regression only runs against mock engines');
  await page.setViewportSize({width:1366,height:900});
  await page.goto(origin + '/chats/new');
  await page.waitForFunction(() => document.querySelector('.chat-layout')?.getAttribute('aria-busy') === 'false' && document.querySelector('.chat-layout')?.getAttribute('data-route') === location.pathname);
  await page.getByRole('textbox', {name:'Image prompt',exact:true}).fill('Source-aware enhancement browser regression');
  const generation = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST');
  await page.getByRole('button', {name:'Generate image',exact:true}).click();
  const generated = await (await generation).json();
  const chat = generated.payload.chat_id;
  await page.waitForURL('**/chats/' + chat);
  await page.getByAltText('Generated image', {exact:true}).waitFor();
  await page.getByAltText('Generated image', {exact:true}).hover();
  await page.getByRole('button', {name:'Enhance',exact:true}).first().click();
  const dialog = page.getByRole('dialog', {name:'Enhance image',exact:true});
  if (!await dialog.getByRole('checkbox', {name:'Automatic image-aware prompt',exact:true}).isChecked()) throw new Error('Automatic planning should default on');
  if (!await dialog.getByRole('checkbox', {name:'Preserve source lighting and colors',exact:true}).isChecked()) throw new Error('Source lighting preservation should default on');
  if (+await dialog.getByRole('slider', {name:'Reconstruction strength',exact:true}).inputValue() !== .65) throw new Error('Unexpected default reconstruction strength');
  const request = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST');
  await dialog.getByRole('button', {name:'Enhance image',exact:true}).click();
  const job = await (await request).json();
  if (!job.payload.adaptive_prompt) throw new Error('Automatic flag missing from request');
  if (!job.payload.preserve_tone) throw new Error('Lighting preservation flag missing');
  if (job.payload.chat_id !== chat) throw new Error('Enhancement escaped its original conversation');
  let complete = false;
  for (let index = 0; index < 300; index++) {
    const current = await (await page.request.get(origin + '/api/jobs/' + job.id)).json();
    if (current.status === 'completed') { complete = true; break; }
    if (['failed','cancelled','interrupted'].includes(current.status)) throw new Error(JSON.stringify(current));
    await page.waitForTimeout(100);
  }
  if (!complete) throw new Error('Enhancement job did not complete');
  const saved = await (await page.request.get(origin + '/api/chats/' + chat)).json();
  const enhanced = saved.turns.find(turn => turn.diagnostics?.enhancement_plan);
  if (!enhanced) throw new Error(JSON.stringify({chat, job:job.id, turns:saved.turns.map(turn => ({id:turn.id,text:turn.text,keys:Object.keys(turn)}))}));
  if (enhanced.processing_passes !== 1 || enhanced.images.length !== 1) throw new Error('One-pass enhancement created multiple outputs');
  if (!enhanced.reconstruction_prompt.includes('Do not invent freckles')) throw new Error('Preservation guard not applied');
  if (!enhanced.diagnostics.enhancement_plan.analysis.mock_not_quality_evidence) throw new Error('Mock analysis not labeled');
  if (enhanced.diagnostics.enhancement_plan.version !== 'scene-profiles-v1') throw new Error('Old skin-first planner is still active');
  if (enhanced.diagnostics.enhancement_plan.profile_id !== 'general') throw new Error('Mock preview must use a labeled fallback, not fabricated recognition');
  if (!enhanced.diagnostics.content_review) throw new Error('Scene content review missing');
  if (!enhanced.diagnostics.tone_anchor.enabled) throw new Error('Tone preservation was not applied');
  if (enhanced.diagnostics.enhancement_review.quality_verified) throw new Error('Pixel changes must not claim verified quality');
  await page.reload();
  await page.getByAltText('Generated image', {exact:true}).last().click();
  await page.getByText('Enhancement analysis', {exact:true}).click();
  await page.getByText('Local preview simulation', {exact:true}).waitFor();
  await page.getByText('General-Purpose Fallback', {exact:true}).waitFor();
  if (await page.getByText('skin texture', {exact:true}).count()) throw new Error('Old skin-centric analysis is still displayed');
  await page.getByText('Content review: mock', {exact:true}).waitFor();
  await page.getByText('Applied reconstruction prompt', {exact:true}).click();
  for (const width of [1366,390,320]) {
    await page.setViewportSize({width,height:900});
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Enhancement analysis overflows viewport');
    await page.screenshot({path:`output/playwright/source-aware-enhancement-${width}.png`});
  }
  await page.keyboard.press('Escape');
  await page.setViewportSize({width:1366,height:900});
  return {passed:true,checks:['automatic default','conservative strength','one output','applied prompt saved','analysis saved','reload provenance','desktop/mobile view']};
}
