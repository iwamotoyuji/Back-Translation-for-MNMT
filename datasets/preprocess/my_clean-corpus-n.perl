#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

# $Id: clean-corpus-n.perl 3633 2010-10-21 09:49:27Z phkoehn $
use warnings;
use strict;
use Getopt::Long;
my $help;
my $lc = 0; # lowercase the corpus?
my $ignore_ratio = 0;
my $ignore_xml = 0;
my $enc = "utf8"; # encoding of the input and output files
    # set to anything else you wish, but I have not tested it yet
my $max_word_length = 1000; # any segment with a word (or factor) exceeding this length in chars
    # is discarded; motivated by symal.cpp, which has its own such parameter (hardcoded to 1000)
    # and crashes if it encounters a word that exceeds it
my $ratio = 9;
my $tgt_eq_img = 0;

GetOptions(
  "help" => \$help,
  "lowercase|lc" => \$lc,
  "encoding=s" => \$enc,
  "ratio=f" => \$ratio,
  "ignore-ratio" => \$ignore_ratio,
  "ignore-xml" => \$ignore_xml,
  "tgt_eq_img" => \$tgt_eq_img,
  "max-word-length|mwl=s" => \$max_word_length
) or exit(1);

if (scalar(@ARGV) < 8 || $help) {
    print "syntax: clean-corpus-n-4.perl [-ratio n] bpe-corpus corpus l1 l2 clean-corpus min max [lines retained file]\n";
    exit;
}

my $bpe_corpus = $ARGV[0];
my $tok_corpus = $ARGV[1];
my $bef_name = $ARGV[2];
my $aft_name =  $ARGV[3];
my $l1 = $ARGV[4];
my $l2 = $ARGV[5];
my $min = $ARGV[6];
my $max = $ARGV[7];

my $linesRetainedFile = "";
if (scalar(@ARGV) > 8) {
	$linesRetainedFile = $ARGV[8];
	open(LINES_RETAINED,">$linesRetainedFile") or die "Can't write $linesRetainedFile";
}

print STDERR "clean-corpus-n-4.perl: processing $bpe_corpus/$bef_name.$l1 $tok_corpus/$bef_name.$l1 & .$l2 to $aft_name, cutoff $min-$max, ratio $ratio\n";

my $opn = undef;
my $bpe_l1input = "$bpe_corpus/$bef_name.$l1";
if (-e $bpe_l1input) {
  $opn = $bpe_l1input;
} elsif (-e $bpe_l1input.".gz") {
  $opn = "gunzip -c $bpe_l1input.gz |";
} else {
    die "Error: $bpe_l1input does not exist";
}
open(F,$opn) or die "Can't open '$opn'";

$opn = undef;
my $bpe_l2input = "$bpe_corpus/$bef_name.$l2";
if (-e $bpe_l2input) {
  $opn = $bpe_l2input;
} elsif (-e $bpe_l2input.".gz") {
  $opn = "gunzip -c $bpe_l2input.gz |";
} else  {
 die "Error: $bpe_l2input does not exist";
}
open(E,$opn) or die "Can't open '$opn'";

$opn = undef;
my $tok_l1input = "$tok_corpus/$bef_name.$l1";
if (-e $tok_l1input) {
  $opn = $tok_l1input;
} elsif (-e $tok_l1input.".gz") {
  $opn = "gunzip -c $tok_l1input.gz |";
} else  {
 die "Error: $tok_l1input does not exist";
}
open(D,$opn) or die "Can't open '$opn'";

$opn = undef;
my $tok_l2input = "$tok_corpus/$bef_name.$l2";
if (-e $tok_l2input) {
  $opn = $tok_l2input;
} elsif (-e $tok_l2input.".gz") {
  $opn = "gunzip -c $tok_l2input.gz |";
} else  {
 die "Error: $tok_l2input does not exist";
}
open(C,$opn) or die "Can't open '$opn'";

open(FO,">$bpe_corpus/$aft_name.$l1") or die "Can't write $bpe_corpus/$aft_name.$l1";
open(EO,">$bpe_corpus/$aft_name.$l2") or die "Can't write $bpe_corpus/$aft_name.$l2";
open(DO,">$tok_corpus/$aft_name.$l1") or die "Can't write $tok_corpus/$aft_name.$l1";
open(CO,">$tok_corpus/$aft_name.$l2") or die "Can't write $tok_corpus/$aft_name.$l2";

# necessary for proper lowercasing
my $binmode;
if ($enc eq "utf8") {
  $binmode = ":utf8";
} else {
  $binmode = ":encoding($enc)";
}
binmode(F, $binmode);
binmode(E, $binmode);
binmode(D, $binmode);
binmode(C, $binmode);
binmode(FO, $binmode);
binmode(EO, $binmode);
binmode(DO, $binmode);
binmode(CO, $binmode);

my $innr = 0;
my $outnr = 0;
my $factored_flag;
while(my $f = <F>) {
  $innr++;
  print STDERR "." if $innr % 10000 == 0;
  print STDERR "($innr)" if $innr % 100000 == 0;
  my $e = <E>;
  die "$bpe_corpus/$bef_name.$l2 is too short!" if !defined $e;
  my $d = <D>;
  die "$tok_corpus/$bef_name.$l1 is too short!" if !defined $d;
  my $c = <C>;
  die "$tok_corpus/$bef_name.$l2 is too short!" if !defined $c;
  chomp($c);
  chomp($d);
  chomp($e);
  chomp($f);
  if ($innr == 1) {
    $factored_flag = ($c =~ /\|/ || $d =~ /\|/ || $e =~ /\|/ || $f =~ /\|/);
  }

  #if lowercasing, lowercase
  if ($lc) {
    $c = lc($c);
    $d = lc($d);
    $e = lc($e);
    $f = lc($f);
  }

  $c =~ s/\|//g unless $factored_flag;
  $c =~ s/\s+/ /g;
  $c =~ s/^ //;
  $c =~ s/ $//;
  $d =~ s/\|//g unless $factored_flag;
  $d =~ s/\s+/ /g;
  $d =~ s/^ //;
  $d =~ s/ $//;
  $e =~ s/\|//g unless $factored_flag;
  $e =~ s/\s+/ /g;
  $e =~ s/^ //;
  $e =~ s/ $//;
  $f =~ s/\|//g unless $factored_flag;
  $f =~ s/\s+/ /g;
  $f =~ s/^ //;
  $f =~ s/ $//;
  next if $c eq '';
  next if $d eq '';
  next if $e eq '';
  next if $f eq '';

  my $cc = &word_count($c);
  my $dc = &word_count($d);
  my $ec = &word_count($e);
  my $fc = &word_count($f);
  next if !$tgt_eq_img && $cc > $max;
  next if $dc > $max;
  next if !$tgt_eq_img && $ec > $max;
  next if $fc > $max;
  next if !$tgt_eq_img && $cc < $min;
  next if $dc < $min;
  next if !$tgt_eq_img && $ec < $min;
  next if $fc < $min;
  next if !$ignore_ratio && $ec/$fc > $ratio;
  next if !$ignore_ratio && $fc/$ec > $ratio;
  # Skip this segment if any factor is longer than $max_word_length
  my $max_word_length_plus_one = $max_word_length + 1;
  next if $e =~ /[^\s\|]{$max_word_length_plus_one}/;
  next if $f =~ /[^\s\|]{$max_word_length_plus_one}/;

  # An extra check: none of the factors can be blank!
  die "There is a blank factor in $bpe_corpus/$bef_name.$l1 on line $innr: $f"
    if $f =~ /[ \|]\|/;
  die "There is a blank factor in $bpe_corpus/$bef_name.$l2 on line $innr: $e"
    if $e =~ /[ \|]\|/;
  die "There is a blank factor in $tok_corpus/$bef_name.$l1 on line $innr: $d"
    if $d =~ /[ \|]\|/;
  die "There is a blank factor in $tok_corpus/$bef_name.$l2 on line $innr: $c"
    if $c =~ /[ \|]\|/;

  $outnr++;
  print FO $f."\n";
  print EO $e."\n";
  print DO $d."\n";
  print CO $c."\n";

  if ($linesRetainedFile ne "") {
	print LINES_RETAINED $innr."\n";
  }
}

if ($linesRetainedFile ne "") {
  close LINES_RETAINED;
}

print STDERR "\n";
my $e = <E>;
die "$bpe_corpus/$bef_name.$l2 is too long!" if defined $e;
my $d = <D>;
die "$tok_corpus/$bef_name.$l1 is too long!" if defined $d;
my $c = <C>;
die "$tok_corpus/$bef_name.$l2 is too long!" if defined $c;

print STDERR "Input sentences: $innr  Output sentences:  $outnr\n";

sub word_count {
  my ($line) = @_;
  if ($ignore_xml) {
    $line =~ s/<\S[^>]*\S>/ /g;
    $line =~ s/\s+/ /g;
    $line =~ s/^ //g;
    $line =~ s/ $//g;
  }
  my @w = split(/ /,$line);
  return scalar @w;
}
