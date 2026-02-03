# Downloads sarxos webcam-capture JAR into project lib folder
$version = "0.3.12"
$webcamUrl = "https://repo1.maven.org/maven2/com/github/sarxos/webcam-capture/$version/webcam-capture-$version.jar"
$slf4jApiVersion = "1.7.36"
$slf4jApiUrl = "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/$slf4jApiVersion/slf4j-api-$slf4jApiVersion.jar"
$slf4jSimpleVersion = "1.7.36"
$slf4jSimpleUrl = "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/$slf4jSimpleVersion/slf4j-simple-$slf4jSimpleVersion.jar"

$destDir = Join-Path -Path "$(Split-Path -Path $PSScriptRoot -Parent)" -ChildPath "lib"
New-Item -ItemType Directory -Path $destDir -Force | Out-Null

function download($url, $dest) {
    Write-Output "Downloading $url to: $dest"
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -ErrorAction Stop
        Write-Output "Download completed: $dest"
    } catch {
        Write-Error ("Failed to download " + $url + ": " + $_)
        exit 1
    }
}

$webcamDest = Join-Path $destDir "webcam-capture-$version.jar"
$slf4jApiDest = Join-Path $destDir "slf4j-api-$slf4jApiVersion.jar"
$slf4jSimpleDest = Join-Path $destDir "slf4j-simple-$slf4jSimpleVersion.jar"

download $webcamUrl $webcamDest
download $slf4jApiUrl $slf4jApiDest
download $slf4jSimpleUrl $slf4jSimpleDest

