# LocoVR-NF — 実住宅の導線尤度で「家具配置」を採点し、再設計をシミュレートする(C トラック)

**データ: LocoVR / LocoReal**(ICLR 2025, MIT。実人間の目的志向二人歩行 × 実住宅)
**位置づけ**: [Layout-NF](../Layout-NF)(TRUMANS 合成室での `p(居場所｜占有)` 採点・逆設計)を、**実データで置き換え・拡張**する姉妹トラック。同じ Normalizing Flow 骨格(zuko NSF・占有クロップ CNN)を使い、**外的妥当性(未知の実住宅への汎化)と再設計能力(目的地条件の動線再ルーティング)**を一段引き上げる。

作成 2026-07-12。親 `.venv`(torch2.12 / zuko1.6 / MPS)、`uv run --no-project python -m locovrnf.*`。

---

## 一言でわかる結果(フェーズ制・ゲート制、負の結果も正直に報告)

| フェーズ | 何を測ったか | 結果 | 判定 |
|---|---|---|---|
| **C0** | LocoReal 実軌跡が占有マップ上を正しく走り家具を避けるか | p1 waist の**99.8%**が非占有セル上(4レイアウトで家具配置が異なり軌跡も相応に変化) | ✅ 整合 |
| **C1a** | 学習した `p(居場所｜占有)` が実家具を低アフォーダンスと判定するか | 実家具セルの logp が free より低い(**全4レイアウトで符号一致・AUC 0.77**、val NLL −0.27 ≪ 一様1.39) | ✅ 合格 |
| **C1b** | 家具化したセルの実訪問密度は落ちるか(クロスレイアウト自然反実) | 4レイアウトはピクセル整合 → **家具化で訪問密度69倍低下**(モデル不要のデータレベル反実=合成A2の実データ版) | ✅ 合格 |
| **C2 座標** | LocoVR 131住宅を各10m地図に登録できるか(未記載の per-home 原点を復元) | 「人は自由空間を歩く」制約でオフセット探索、**131/131住宅を on-free 中央値100%・平均99.8%で登録** | ✅ 整合 |
| **C2 汎化** | 未知の**実住宅**での密度NLLと家具検出AUC(A4/A7の131実住宅版) | held-out住宅NLL **−0.434 ≈ in-dist −0.399**(gap比1.09)。家具AUCは住宅数10→100で **0.65→0.87** と上昇 | ✅ 汎化成立 |
| **C3 生成** | 目的地条件の軌跡分布が家具変更後の再ルーティングを予測できるか | 生成分布の **minADE 0.387m < A* 0.679m < 直線 0.809m**。家具重なりは直線の**1/3**(0.088 vs 0.269)。家具を差し替えると経路分布が再ルーティング | ✅ 再設計シミュ成立 |
| **C3 注記★** | 経路全体の per-step 尤度で実経路 vs 直線を判別できるか | できない:**直線の方が per-step logp が高い**(4.21 vs 3.22)。ステップ尤度はゴール直進を最尤とするため大域的迂回品質を測れない | ⚠ 局所尤度≠大域導線(A1/A2の教訓の再現) |
| **C5 最適化** | アフォーダンスを目的関数に N個の家具配置を同時最適化(A3の多家具・最適化版) | 空室の foot-traffic 場 `T(x)=exp logp(居場所)` を固定コスト場に、CEM で家具を低traffic の縁へ。**動線妨害 26.7%→14.9%(44%減)**、~2秒 | ✅ 逆設計の最適化成立 |

**C1 の含意**: Layout-NF が合成の家具貼りで示した「配置採点」は、実人間・実家具・実占有でも成立。C1b は THÖR-MAGNI の B8(同一位置での障害物あり/なし=88倍)を、家具の置き換えという形で再現した実データ版。
**C2 の含意**: THÖR(1室)では不可能だった「未知の実空間へ汎化」を131実住宅で実証。TRUMANS A7 が「密度は多様性で改善/検出は早期飽和」だったのに対し、実住宅では**検出AUCも住宅数とともに上昇し続ける**(合成より実データの多様性が効く)。
**C3 の含意**: 静的密度から**目的地条件の軌跡フロー**へ定式化を変え、家具を動かすと動線分布がどう引き直されるかを、実測の4レイアウト(相互 ground truth)で検証。生成は「フローが提案し占有が却下する」占有認識ロールアウト。per-step 尤度ランキングは効かず(正直な負の結果)、主張は生成分布の ADE に置く。

---

## モデル

`locovrnf/model.py` — Layout-NF と同一骨格。

- **`AffordanceFlow`**(C1/C2): `p(訪問位置の2Dオフセット ｜ 48×48占有パッチCNN)`、zuko 条件付き NSF。占有は実マップ(10cm/セルに再サンプル)。
- **`StepFlow`**(C3): `p(次ステップのego変位 ｜ 前方占有パッチ, ego座標のゴールベクトル, 速度)`、自己回帰。start→goalへロールアウトして**経路の分布**を生成=再設計シミュレータ。
- **`optimize.py`**(C5): アフォーダンスフロー(C1)を目的関数にした CEM。空室の居場所密度を foot-traffic コスト場とし、家具 N 個を「動線を最も妨げない配置」へ同時最適化(A3の逆設計を多家具・最適化へ拡張)。

## データ

- **LocoReal**(物理・実キャンパス室・**家具4配置**×5人×430軌跡・62MB): 同一室で家具だけ差替えた4レイアウト。`p1`=目的志向の歩行者(`p2`は静的レイアウト問題では無視)。各人物 `head/right hand/waist` の `{pos:(T,3)m, pose:(T,3)euler}`。占有 `binary_map/00x.png`(1024², 黒=家具・壁)。座標 `px=(m+5)·1024/10`(公式 `vis_traj.py`)。
- **LocoVR**(VR・**131実住宅**(HM3Dスキャン)・7000+軌跡・2.4GB): 6部位・pose は quat。地図は住宅ごとの10m crop で**世界原点が住宅ごとに異なる**ため、`locovrnf.register` が軌跡の自由空間整合からオフセットを復元(`data/processed/locovr_offsets.json` にキャッシュ)。
- **入手**(gdown、MIT): `uv run --no-project python -m locovrnf.fetch locoreal locovr maps testcode`。`data/` は gitignore(再DL可)。

## 出力(`results/`)

- `c0_coordcheck.{png,json}` — LocoReal 4レイアウトの軌跡×占有整合(99.8%)
- `c1_affordance.{png,json}` — 家具 vs free の logp 分離(AUC 0.77)/ 69倍の訪問密度低下 / 訪問密度×家具
- `c2_register.png` — 131住宅の座標登録品質(中央値100%)+ 登録例
- `c2_scale.{png,json}` — 未知住宅での NLL/AUC スケール曲線
- `c3_traj.json` / `c3_redesign.png` — 生成分布 vs 実経路 + 家具差替えによる再ルーティング
- `c5_optimize.{png,json}` — foot-traffic 場 / naive vs CEM最適化の家具配置(動線妨害44%減)

## 再現

```bash
uv run --no-project python -m locovrnf.fetch locoreal locovr maps testcode  # DL(~2.5GB)
uv run --no-project python -m locovrnf.coordcheck    # C0 LocoReal 座標整合
uv run --no-project python -m locovrnf.affordance    # C1 実家具回避の採点(~15s)
uv run --no-project python -m locovrnf.register      # C2 LocoVR 131住宅の座標登録(~5s)
uv run --no-project python -m locovrnf.crossspace --epochs 10  # C2 cross-space汎化(~2.5min)
uv run --no-project python -m locovrnf.traj          # C3 目的地条件の軌跡生成=再設計(~30s)
uv run --no-project python -m locovrnf.optimize      # C5 家具配置の最適化(CEM, ~2s)
```

## 制約 / 次の一手

- **LocoVR は VR**: 実歩行と微差の可能性。→ C4(LocoReal 物理レイアウトを VR 学習モデルの物理テストセットに)で担保する設計(未着手)。
- **占有の意味**: binary map は家具+壁=黒(通れない障害物)。TRUMANS の「家具に近づいて使う」滞在とは逆(単独障害物を避けて通り抜け)——両データの結論は設定差込みで述べる。
- **C3 の生成**: ゴール直近で実人間ほど障害物を避けきれず(重なり 0.088 vs 実人間 0.005)。占有認識ロールアウトで直線比1/3までは抑制。複数家具の同時最適化・社会的回避(p2条件化)へ拡張余地。
- 継続保持: **THÖR-MAGNI**(同一室の因果検証 B8)・**TRUMANS**(滞在主体データの対照)。

KAN-NF ハブ: [kankusakabe.github.io/KAN-NF](https://kankusakabe.github.io/KAN-NF/)
