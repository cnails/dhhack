$file = $args[0]
$time = $args[1]

$ind = $file.IndexOf(".")
$out = $file.Substring(0, $ind)
$file_out = $out + "_cens.mp3"


foreach($line in Get-Content $time) {
	$a, $b, $c = $line.Split(" ")

	$ind = $a.IndexOf(".")
	$sec = $a.Substring(0, $ind)
	$ml = $a.Substring($ind + 1)
	$s_a = New-TimeSpan -Seconds $sec
	$s_a = $s_a + [TimeSpan]::FromMilliseconds(10 * $ml)
	$t_a = ("{0:hh\:mm\:ss\.ff}" -f $s_a)
	echo $t_a

	$ind = $b.IndexOf(".")
	$sec = $b.Substring(0, $ind)
	$ml = $b.Substring($ind + 1)
	$s_b = New-TimeSpan -Seconds $sec
	$s_b = $s_b + [TimeSpan]::FromMilliseconds(10 * $ml)
	$t_b = ("{0:hh\:mm\:ss\.ff}" -f $s_b)
	echo $t_b
	$d = (("{0:ss}" -f $s_b) + ("{0:fff}" -f $s_b))

	$s_c = $s_a + $s_b
	$t_c = ("{0:hh\:mm\:ss\.ff}" -f $s_c)
	echo $t_c

	ffmpeg -y -t $t_a -i $file trim_1.mp3
	ffmpeg -y -ss $t_c -i $file trim_2.mp3
	ffmpeg -y -i trim_2.mp3 -af "adelay=$d|$d" trim_3.mp3
	ffmpeg -y -i "concat:trim_1.mp3|trim_3.mp3" -acodec copy $file_out
	$file = $file_out
}
