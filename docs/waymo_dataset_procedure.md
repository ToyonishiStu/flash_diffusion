前提条件
# Google Cloud SDK インストール済みであること
# インストールスクリプトで一括インストール
curl https://sdk.cloud.google.com | bash

# シェルを再起動
exec -l $SHELL

# 初期化（ブラウザでGoogleログイン画面が開く）
gcloud init

または apt 経由：

# リポジトリ追加
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt
cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

sudo apt-get update && sudo apt-get install -y google-cloud-cli

# 初期化
gcloud init

インストール確認

gcloud --version
gsutil --version

WSL2での注意点

gcloud init 時にブラウザが自動で開かない場合は：
gcloud auth login --no-launch-browser

#ダウンロード手順
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# データ保存先ディレクトリ作成
mkdir -p /workspaces/toyot/flash_diffusion/data/training
mkdir -p /workspaces/toyot/flash_diffusion/data/validation

Waymo Open Datasetへのアクセスはあらかじめ https://waymo.com/open で利用申請・承認が必要です。

---
データセットバージョンとVirtual Range Imageについて

Waymo v2 (gs://waymo_open_dataset_v_2_0_0/) のコンポーネント構成：

gs://waymo_open_dataset_v_2_0_0/
├── training/
│   ├── lidar/                        ← raw点群
│   ├── lidar_range_image/            ← range image (real sensor)
│   └── ...

Virtual range imageは gs://waymo_open_dataset_v_1_4_3/（v1, tfrecord形式）に含まれています。
v1のtfrecordには ri_return1 / ri_return2（realおよびvirtual）が埋め込まれています。super
resolutionの先行研究でもv1 tfrecordが主流です。

---
開発デバイス（training/validation 各1シーン）

  mkdir -p /workspaces/toyot/flash_diffusion/data/training
  mkdir -p /workspaces/toyot/flash_diffusion/data/validation

  gcloud storage cp \
    "gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-10017090168044687777_6380_000_6
  400_000_with_camera_labels.tfrecord" \
    /workspaces/toyot/flash_diffusion/data/training/

  gcloud storage cp \
    "gs://waymo_open_dataset_v_1_4_3/individual_files/validation/segment-10203656353524179475_7625_000
  _7645_000_with_camera_labels.tfrecord" \
    /workspaces/toyot/flash_diffusion/data/validation/

  ---
  学習デバイス（training全件 + validation 50シーン）

  mkdir -p /workspaces/toyot/flash_diffusion/data/training
  mkdir -p /workspaces/toyot/flash_diffusion/data/validation

  # training 全件
  gcloud storage cp \
    "gs://waymo_open_dataset_v_1_4_3/individual_files/training/*.tfrecord" \
    /workspaces/toyot/flash_diffusion/data/training/

  # validation 50シーン
  gcloud storage ls gs://waymo_open_dataset_v_1_4_3/individual_files/validation/ \
    | grep '\.tfrecord' \
    | head -50 \
    | xargs -I{} gcloud storage cp {} /workspaces/toyot/flash_diffusion/data/validation/

---
補足

┌───────────────────────────────┬────────────────────────────────────────────────────────┐
│             項目              │                          内容                          │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Training scenes               │ 798シーン                                              │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Validation scenes             │ 202シーン（うち50を使用）                              │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1シーンあたりのサイズ         │ 約1〜2GB                                               │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Virtual range imageの取り出し │ tfrecordをパースして frame.lasers[0].ri_return1 を読む │
└───────────────────────────────┴────────────────────────────────────────────────────────┘

まず gsutil ls gs://waymo_open_dataset_v_1_4_3/individual_files/training/
でアクセス確認をしてから実行することをお勧めします。
