param(
    [string]$InputFolder = $PSScriptRoot,
    [string]$OutputFolder = (Join-Path $PSScriptRoot 'normalized-holdings'),
    [switch]$AggregateSameSymbolRows
)

Write-Host "--- Diagnostics ---"
Write-Host "Looking in: $InputFolder"
Write-Host "Saving to: $OutputFolder"

if (-not (Test-Path -LiteralPath $OutputFolder)) {
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
    Write-Host "Created output directory."
}

$files = Get-ChildItem -LiteralPath $InputFolder -Filter 'Portfolio_Positions_*.csv'
Write-Host "Found $($files.Count) file(s) matching 'Portfolio_Positions_*.csv'"

$monthMap = @{
    Jan = 1; Feb = 2; Mar = 3; Apr = 4; May = 5; Jun = 6;
    Jul = 7; Aug = 8; Sep = 9; Oct = 10; Nov = 11; Dec = 12
}

function Get-SnapshotDate {
    param([string]$Path)
    $lines = @(Get-Content -LiteralPath $Path)
    $name = Split-Path -Leaf $Path
    
    if ($name -match 'Portfolio_Positions_(?<mon>[A-Za-z]{3})-(?<day>\d{2})-(?<year>\d{4})') {
        $month = $monthMap[$Matches.mon]
        if ($month) { return [datetime]::new([int]$Matches.year, $month, [int]$Matches.day).ToString('yyyy-MM-dd') }
    }
    throw "Unable to determine snapshot date from filename: $name"
}

function Get-RowValue {
    param([psobject]$Row, [string[]]$Names, [string]$Default = '')
    foreach ($name in $Names) {
        foreach ($property in $Row.PSObject.Properties) {
            if ($property.Name.Trim() -ieq $name) {
                if ($null -ne $property.Value) { return ([string]$property.Value).Trim() }
            }
        }
    }
    return $Default
}

function Format-File {
    param([string]$Path, [string]$DateValue)
    $lines = @(Get-Content -LiteralPath $Path)
    if ($lines.Count -lt 2) { return @() }
    
    $parsedRows = $lines | ConvertFrom-Csv
    
    $results = foreach ($row in $parsedRows) {
        # Create a clean object with explicit names
        [pscustomobject]@{
            'Date' = $DateValue
            'Symbol' = $row.Symbol
            'Description' = $row.Description
            # Remove '$' and ',' from currency so it imports as a number
            'Current_Value' = ($row.'Current value' -replace '[$,]', '') 
        }
    }
    return $results
}

# Execution
foreach ($file in $files) {
    try {
        $snapshotDate = Get-SnapshotDate -Path $file.FullName
        Write-Host "Processing: $($file.Name) (Date: $snapshotDate)"
        $rows = Format-File -Path $file.FullName -DateValue $snapshotDate
        if ($rows) {
            $outputPath = Join-Path $OutputFolder ("Portfolio_Positions_{0}.csv" -f $snapshotDate)
            $rows | Export-Csv -LiteralPath $outputPath -NoTypeInformation -Encoding utf8
            Write-Host "Successfully exported to $outputPath"
        } else {
            Write-Host "Warning: No data rows found in $($file.Name)"
        }
    } catch {
        Write-Host "Error processing $($file.Name): $_"
    }
}