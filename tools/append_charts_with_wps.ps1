param(
  [string]$InputDocx = "C:\Users\29189\Desktop\毕业设计\图表.docx",
  [string]$OutputDocx = "C:\Users\29189\Desktop\毕业设计\图表-制图版.docx",
  [string]$ChartDir = "C:\Users\29189\Desktop\毕业设计\制图输出"
)

$ErrorActionPreference = "Stop"

Copy-Item -LiteralPath $InputDocx -Destination $OutputDocx -Force

$metrics = Invoke-RestMethod -Uri "http://127.0.0.1:8765/metrics" -TimeoutSec 10
$benchmarkRows = @()
if ($metrics.last_benchmark -and $metrics.last_benchmark.rows) {
  $benchmarkRows = @($metrics.last_benchmark.rows)
}
$marketRows = @()
if ($metrics.database_comparison) {
  $marketRows = @($metrics.database_comparison | Select-Object -First 10)
}

$app = New-Object -ComObject "KWPS.Application"
$app.Visible = $false
try { $app.DisplayAlerts = 0 } catch {}

$doc = $app.Documents.Open($OutputDocx)
$sel = $app.Selection
$sel.EndKey(6) | Out-Null
$sel.InsertBreak(7)

function Set-Font([int]$Size, [bool]$Bold = $false) {
  $script:sel.Font.Name = "微软雅黑"
  $script:sel.Font.NameFarEast = "微软雅黑"
  $script:sel.Font.Size = $Size
  $script:sel.Font.Bold = $(if ($Bold) { 1 } else { 0 })
}

function Add-Text([string]$Text, [int]$Size = 11, [bool]$Bold = $false) {
  Set-Font $Size $Bold
  $script:sel.TypeText($Text)
  $script:sel.TypeParagraph()
}

function Add-Heading([string]$Text, [int]$Level = 1) {
  $size = $(if ($Level -eq 1) { 18 } elseif ($Level -eq 2) { 15 } else { 13 })
  Add-Text $Text $size $true
}

function Add-PictureBlock([string]$Caption, [string]$FileName) {
  $path = Join-Path $ChartDir $FileName
  Add-Text $Caption 12 $true
  if (Test-Path $path) {
    $shape = $script:sel.InlineShapes.AddPicture($path, $false, $true)
    if ($shape.Width -gt 500) { $shape.Width = 500 }
    $script:sel.TypeParagraph()
    $script:sel.TypeParagraph()
  } else {
    Add-Text "图片文件未找到：$path" 11 $false
  }
}

function Add-TableBlock([string]$Title, $Rows) {
  Add-Text $Title 12 $true
  if (-not $Rows -or $Rows.Count -eq 0) {
    Add-Text "暂无数据。" 11 $false
    return
  }
  $rowCount = $Rows.Count
  $colCount = $Rows[0].Count
  $range = $script:sel.Range
  $table = $script:doc.Tables.Add($range, $rowCount, $colCount)
  try { $table.Borders.Enable = 1 } catch {}
  for ($r = 1; $r -le $rowCount; $r++) {
    for ($c = 1; $c -le $colCount; $c++) {
      $cell = $table.Cell($r, $c)
      $cell.Range.Text = [string]$Rows[$r - 1][$c - 1]
      $cell.Range.Font.NameFarEast = "微软雅黑"
      $cell.Range.Font.Name = "微软雅黑"
      $cell.Range.Font.Size = $(if ($r -eq 1) { 10 } else { 9 })
      if ($r -eq 1) {
        $cell.Range.Font.Bold = 1
        try { $cell.Shading.BackgroundPatternColor = 15132390 } catch {}
      }
      try { $cell.VerticalAlignment = 1 } catch {}
    }
  }
  try { $table.AutoFitBehavior(2) | Out-Null } catch {}
  $after = $table.Range
  $after.Collapse(0)
  $after.Select()
  $script:sel.TypeParagraph()
}

Add-Heading "六、制图成品（可直接用于论文和答辩）" 1
Add-Text "以下图表根据《图表.docx》的清单生成，可在 WPS 中继续调整颜色、大小和位置。实验图的数据来自当前系统监控页与 benchmark 输出。" 11 $false

Add-Heading "6.1 系统设计类图" 2
Add-PictureBlock "图1-1 课题技术路线图" "图1-1_课题技术路线图.png"
Add-PictureBlock "图2-1 近似最近邻检索方法分类" "图2-1_近似最近邻检索方法分类.png"
Add-PictureBlock "图4-1 系统总体架构图" "图4-1_系统总体架构图.png"
Add-PictureBlock "图4-2 系统业务流程图" "图4-2_系统业务流程图.png"

$fieldRows = [System.Collections.ArrayList]::new()
[void]$fieldRows.Add([object[]]@("字段名", "类型", "说明"))
[void]$fieldRows.Add([object[]]@("item_id", "INT64", "商品唯一编号，主键"))
[void]$fieldRows.Add([object[]]@("category_id", "INT64", "商品类目编号，用于结构化过滤"))
[void]$fieldRows.Add([object[]]@("category_name", "VARCHAR", "商品类目名称"))
[void]$fieldRows.Add([object[]]@("price", "FLOAT", "商品价格，用于价格区间筛选"))
[void]$fieldRows.Add([object[]]@("product_name", "VARCHAR", "商品名称"))
[void]$fieldRows.Add([object[]]@("brand", "VARCHAR", "品牌信息"))
[void]$fieldRows.Add([object[]]@("product_type", "VARCHAR", "商品种类"))
[void]$fieldRows.Add([object[]]@("size", "VARCHAR", "规格或尺寸"))
[void]$fieldRows.Add([object[]]@("purpose", "VARCHAR", "使用场景或用途"))
[void]$fieldRows.Add([object[]]@("target_group", "VARCHAR", "目标消费群体"))
[void]$fieldRows.Add([object[]]@("market", "VARCHAR", "市场定位"))
[void]$fieldRows.Add([object[]]@("embedding", "FLOAT_VECTOR(128)", "商品语义向量，用于近似检索"))
Add-TableBlock "表5-1 商品向量集合字段结构" $fieldRows
Add-PictureBlock "图5-1 核心模块调用关系" "图5-1_核心模块调用关系.png"

Add-Heading "6.2 实验结果图表" 2
Add-PictureBlock "图6-1 不同数据规模下平均检索延迟" "图6-1_不同数据规模下平均检索延迟.png"
Add-PictureBlock "图6-2 不同数据规模下 Recall@K" "图6-2_不同数据规模下RecallK.png"

$benchTable = [System.Collections.ArrayList]::new()
[void]$benchTable.Add([object[]]@("数据规模", "插入耗时(ms)", "平均检索(ms)", "Recall@K", "MRR@K", "NDCG@K", "更新耗时(ms)", "删除耗时(ms)", "备注"))
foreach ($row in $benchmarkRows) {
  $note = switch ([int]$row.dataset_size) {
    100 { "小规模功能验证" }
    1000 { "基础性能展示" }
    5000 { "中等规模压力" }
    10000 { "较大规模演示" }
    default { "benchmark 实测" }
  }
  [void]$benchTable.Add([object[]]@(
    [string]$row.dataset_size,
    [string]$row.insert_ms,
    [string]$row.avg_search_ms,
    [string]$row.avg_recall_at_k,
    [string]$row.avg_mrr_at_k,
    [string]$row.avg_ndcg_at_k,
    [string]$row.update_ms,
    [string]$row.delete_ms,
    $note
  ))
}
Add-TableBlock "表6-0 实验指标记录表（benchmark 实测）" $benchTable

$funcRows = [System.Collections.ArrayList]::new()
[void]$funcRows.Add([object[]]@("测试项", "测试内容", "结果"))
[void]$funcRows.Add([object[]]@("集合初始化", "创建或重建 Milvus 商品向量集合", "通过"))
[void]$funcRows.Add([object[]]@("批量生成", "按指定数量生成 100-1000000 条商品数据", "通过"))
[void]$funcRows.Add([object[]]@("商品写入", "写入商品属性和 embedding 向量", "通过"))
[void]$funcRows.Add([object[]]@("自然语言检索", "输入商品需求并返回 Top-K 结果", "通过"))
[void]$funcRows.Add([object[]]@("过滤检索", "按类目、价格区间组合过滤", "通过"))
[void]$funcRows.Add([object[]]@("动态更新", "支持新增、删除与 upsert", "通过"))
[void]$funcRows.Add([object[]]@("实时监控", "监测任务进度、延迟、Recall@K、MRR@K、NDCG@K 和对比指标", "通过"))
[void]$funcRows.Add([object[]]@("同数据集基线", "接入 Local ANN 与精确顺序扫描进行实测", "通过"))
Add-TableBlock "表6-1 系统功能测试结果" $funcRows

function Short-MarketStatus($Text) {
  if ([string]::IsNullOrWhiteSpace($Text)) { return "-" }
  if ($Text -like "未配置外部连接*") { return "未配置连接，待同数据集实测" }
  return $Text
}

function Short-MarketFit($Name, $Text) {
  switch -Wildcard ($Name) {
    "当前系统*" { return "本课题实测对象" }
    "Local ANN*" { return "本地近似检索基线" }
    "精确顺序扫描*" { return "精确扫描性能基线" }
    "Pinecone" { return "云上语义检索与推荐" }
    "Zilliz Cloud" { return "Milvus 生产化托管方案" }
    "Qdrant" { return "轻量部署、过滤丰富" }
    "Weaviate" { return "语义搜索与关键词混合" }
    "Elasticsearch*" { return "搜索体系升级语义检索" }
    "PostgreSQL*" { return "中小规模、强事务场景" }
    "Redis Vector" { return "低延迟召回与缓存层" }
    default { return $Text }
  }
}

$marketTable = [System.Collections.ArrayList]::new()
[void]$marketTable.Add([object[]]@("方案", "类型", "实测/接入状态", "适用性"))
foreach ($row in $marketRows) {
  [void]$marketTable.Add([object[]]@(
    [string]$row.name,
    [string]$row.type,
    [string](Short-MarketStatus $row.measured_summary),
    [string](Short-MarketFit $row.name $row.fit)
  ))
}
Add-TableBlock "表6-2 主流检索方案能力与实测状态对比" $marketTable
Add-PictureBlock "图6-3 实时监控面板截图" "图6-3_实时监控面板截图.png"

$doc.Save()
$doc.Close()
$app.Quit()

Write-Output $OutputDocx
