# 每日开盘信号邮件 — 安装说明

## 传什么

```
.github/workflows/daily-signal.yml     ← 定时任务(隐藏目录,见下面的传法)
signal/daily_signal.py
signal/strategy_core.py
signal/leverage_backtest_cap3.py
```

**`daily_state.json` 不用再传一份** —— 脚本会自动去仓库根目录找你为网页传的那一份。
两处各存一份迟早会不一致,所以只留一份。

## 隐藏目录怎么传:不要拖拽,用网页新建

`.github` 以点开头,macOS 访达默认不显示,拖不进去。**用 GitHub 网页直接建:**

1. 打开仓库首页 → **Add file** → **Create new file**
2. 文件名那一栏,**整串路径打进去**(带斜杠):
   ```
   .github/workflows/daily-signal.yml
   ```
   打到每个 `/` 时 GitHub 会自动建出目录,你会看到路径变成一级级的
3. 把 `daily-signal.yml` 的内容整个粘进编辑框
4. 页面最下面 **Commit new file**

`signal/` 那三个 .py 同理,或者用 **Add file → Upload files** 直接把整个 `signal` 文件夹拖进去(这个目录不隐藏,拖得动)。

## 设置 Secrets

仓库 → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| 名称 | 填什么 |
|---|---|
| `SMTP_HOST` | 如 `smtp.qq.com` / `smtp.163.com` / `smtp.gmail.com` |
| `SMTP_PORT` | `465`(SSL)或 `587`(STARTTLS) |
| `SMTP_USER` | 发件邮箱地址 |
| `SMTP_PASS` | **授权码,不是登录密码**(QQ/163 在邮箱设置里生成,Gmail 用 App Password) |
| `MAIL_TO` | 收件地址,不填就发给自己 |
| `POSITIONS_GIST_RAW_URL` | 可选。网页同步持仓那个 Gist 的 raw 链接,填了邮件才带止盈价 |

这些只有你能设,仓库代码里没有任何凭据。

## 先手动跑一次

**Actions** 标签页 → 左边选 **每日开盘信号** → 右边 **Run workflow** → 绿色按钮。

跑完点进去看日志。成功会显示 `[ok] 已发送至 ...`。失败最常见的原因是 `SMTP_PASS` 填了登录密码而不是授权码。

## 时间

`cron: "12 1 * * 1-5"` = **UTC 01:12 = 北京 09:12**,周一到周五。

不是 9:25,因为 GitHub 定时任务会漂 5~30 分钟、偶尔跳过。所以提早起跑,然后轮询等开盘价(最多 15 分钟)。**迟到不等于出错**:开盘价 9:25 定下后一整天不变,晚发的邮件里挂单价照样对。

改时间就改 cron,注意**它是 UTC,要减 8 小时**。

## 休市

中国节假日写不进 cron。任务会先查"今天商品日盘开了没",没开就跳过。查的是数据不是节假日表,调休也能应付。

## 维护

`daily_state.json` 里带着模型的历史输入,过期了模型就少训练数据(网页会在"日盘开盘价覆盖到"那行变红提醒)。每周本地跑一次:

```bash
python3 fetch_dayopen.py && python3 export_state.py
```

然后把新的 `daily_state.json` 传到**仓库根目录**(网页和邮件共用这一份)。
