# Annotation protocol (final)

For people labeling leaf squares in the tiles annotator.

You describe **one square**. You do not score the plant 1 to 5. You do not decide whether the plant is a 3 or a 4. That number is computed later from all of the young-growth squares.

Chilli thrips are almost never visible. You are marking **damage on tissue**.

---

## What to open

1. Start the app and open http://127.0.0.1:8765
2. Type your name on Home.
3. First time on this set: **Local tiles**, upload or paste the tiled folder (`tiles_foliage.csv` and `foliage_tiles`). That is the 36 plants already cut into squares.
4. Later: **Continue last session** on this same computer.

A finished square is saved on this computer as soon as the last required click happens. You do not export after each plant. The next plant opens by itself. Export CSV (or Export to GPU) is only when you want a copy.

---

## The two pictures

- **Left:** the square you mark. The buttons apply only to this square.
- **Right:** the whole plant, current square outlined. Use this to see if you are at the top (usually new growth), lower canopy (usually old leaves), or on the white tube.

If a leaf is not inside the outlined square, ignore it.

---

## Order of questions

**Every square:** What tissue is this?

1. New growth  
2. Old leaves  
3. Not a leaf  

**Old leaves** or **Not a leaf:** stop. The square is saved as skip. Do not mark injured. Do not mark curl.

**New growth only:**

1. Healthy, injured, or skip  
2. If healthy or injured: curl yes or curl no  

The square is not saved until that chain is finished.

---

## 1. New growth

Young flush: soft, recently opened leaves at the top of the plant.

- Thinner and more tender than the old canopy.
- Often lime, yellow-green, or pale / silver if already hit.
- In the whole-plant picture they sit high.

**Pale or silvered young leaves are still new growth.** Do not send them to Not a leaf because they are not dark green. Those tips are often the damage we care about.

If the square mixes young leaves with tube or old leaf: choose New growth when you can still judge a real patch of young flush. If the young bit is a sliver you cannot judge, choose Old leaves or Not a leaf.

---

## 2. Old leaves

Mature canopy or old wood: thicker, darker, leathery leaves, often lower or outer, sometimes woody stem.

Old leaves may be bronze, spotted, curled, or worn. **That is not scored.** Mark **Old leaves** and stop. Do not mark injured. The official plant score is on new flush only. If you mark old canopy injured, a plant with a clean top and a beat-up bottom will look worse than it should.

If you cannot tell young from old, use the whole-plant picture. Hardened mid-canopy is old. Soft current top is new.

---

## 3. Not a leaf

Not foliage we should score:

- White grow tube  
- Pot, tray, tag, stake  
- Mulch, soil, sky  
- Empty blur or glare with no readable leaf  
- Almost only stem or tube wall  

If the square is clearly young leaves and the tube is only an edge, use New growth. If the readable content is the tube, use Not a leaf.

---

## New growth: healthy, injured, or skip

This is only about **thrips-type damage on the young tissue in this square**. It is not a 1 to 5.

### Healthy

The young leaves in the square look clean for thrips.

- Normal green or lime new leaves.
- **Lime crinkle with no bronze and no cup = healthy.** Some young blueberry leaves are puckered or lime without looking scraped or silvered.
- You may still say curl yes if the shape is cupped or crinkled. Healthy + curl yes is allowed.

Do not mark healthy because the small whole-plant picture “looks okay” if this square itself is bronze or silver.

### Injured

The young leaves show thrips-type damage:

- Bronze or silver sheen  
- Cupped young leaves  
- Crinkle that comes with that scraped or metallic look  
- Dark or dying young tips that fit with flush damage  

**Mixed new growth = injured.** If part of the young flush in the square is hurt and part is clean, mark injured. Thirty percent hurt or seventy percent hurt is the same button. Do not write a percent. Do not mark healthy because “most of it looks fine.”

Injured with **no curl** is still injured. It is not a half mark.

Do not mark injured for old-leaf wear (that should have been Old leaves) or for tube and dirt (Not a leaf).

### Skip

The square is new growth but you cannot honestly tell (heavy blur, tiny scrap, glare that hides the surface). Skip should be uncommon. If you can see the leaf surface, choose healthy or injured. Do not use skip because the plant feels like a 3 instead of a 4.

---

## Curl (new growth, after healthy or injured)

Curl is **shape**, not color.

- **Yes:** young growth is cupped, rolled, twisted, or crinkled so the blade is not flat.
- **No:** young blades are basically flat, even if they are bronze or silver.

Allowed combinations:

| Injury | Curl | Meaning |
|---|---|---|
| Healthy | No | Clean, flat young leaf |
| Healthy | Yes | Clean color, but cupped or crinkled (lime crinkle with no bronze is this) |
| Injured | No | Bronze or silver, but flat |
| Injured | Yes | Hurt and deformed |

You must pick yes or no. The app will not save healthy or injured without curl.

---

## Do not turn a square into a plant score

You will see many outer young squares that look injured and curled. Mark what is in the square. Do not leave some injured squares healthy so the plant “looks like a 3.”

A 3 versus a 4 is **not** your job. It is computed later from how many young squares were injured, where they sat (the top counts more), how many were curled, and how those tiles look. If almost every young square on this set is injured, that is all right. Do not invent healthy marks to make the count look nicer.

---

## Quick decisions

| What you see | Mark |
|---|---|
| Soft young top leaves, clean | New growth → healthy → curl |
| Young leaves, bronze / silver / thrips cup or crinkle | New growth → injured → curl |
| Young leaves, some hurt and some clean | New growth → injured → curl |
| Young leaves, lime crinkle, no bronze, no cup | New growth → healthy → curl (yes if crinkled) |
| Young leaves, hurt but flat | New growth → injured → curl no |
| Old hard leaves, even if bronze or curled | Old leaves |
| Tube, pot, tag, sky, dirt, empty | Not a leaf |
| Pale silver young tips | New growth, then healthy or injured from the surface |
| Cannot see the young surface | New growth → skip |

---

## Keys

- `1` `2` `3` — tissue, then (if new growth) healthy / injured / skip  
- `Y` / `N` — curl  
- `Z` — undo last save  
- `Esc` — clear an unfinished square  
- Left / right arrows — move along this plant  

Wrong save: `Z`, or open Review, click the square, fix it. After you save the fix you return to Review.

---

## Saving

Old leaves and not a leaf save on that one click. New growth saves when you finish skip, or when you finish healthy or injured plus curl.

Close the browser whenever you want. Next time: same address, your name, **Continue last session**. Stay on this computer. A new folder upload starts a new session and does not add marks to the old one.

You do not need to export after each plant. All 36 photos are one session. When a plant’s last square is done, the next plant opens.

---

## Do not

- Do not score a square 1 to 5.  
- Do not mark old leaves injured.  
- Do not mark tube or dirt injured.  
- Do not treat pale silver flush as not a leaf.  
- Do not estimate a percent. Mixed young flush is injured.  
- Do not hold back an injured mark because the plant should be a 3.  
- Do not put pictures, labels, or the GPU password on GitHub.  
