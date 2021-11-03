


<?php
$source_pdf="result.pdf";
$output_folder="MyFolder";

if (!file_exists($output_folder)) { mkdir($output_folder, 0777, true);}
$a= passthru("pdftohtml $source_pdf $output_folder/new_file_name",$b);
var_dump($a);
?>



