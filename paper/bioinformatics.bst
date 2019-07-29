% BibTex `bioinformatics' bibliography style
% http://bioinformatics.oupjournals.org/
% http://www.lecb.ncifcrf.gov/~toms/latex.html
%
%    version = 1.02 of bioinformatics.bst 2004 Sep 23
% 2004 Sep 23, 1.02: modified by Allan A. Sioson (see notes below)
% 2001 Apr  2, 1.01: origin by Bruce Shapiro from jmb.bst
%
% FROM:
% BibTex `jmb' bibliography style
% version = 1.28 of jmb.bst 1992 November 2
%
% Thomas Schneider
% National Cancer Institute
% Laboratory of Mathematical Biology
% Frederick, Maryland  21702-1013
% toms@ncifcrf.gov
%
% This file is available by anonymous ftp from ncifcrf.gov in pub/delila.
%
% TITLES can be turned on and off!
%      Just nocite the reference TitlesOn in the paper (i.e. \nocite{TitlesOn})
%      and have a bibliography article in your database with that cite key!
% Without titles is Journal of Molecular Biology;
% With titles is Journal of Theoretical Biology.
%
% WARNING: Since I use mostly article, book and inproceedings, these
% are formatted pretty closely to the Journal of Theoretical Biology style
% while other things are neglected.  Also, I can't guarantee that the style
% is exactly right.
%
% The following documentation is identical from the source of jmb.bst,
% which was the apalike.bst taken from the Clarkson archive on 1989 June 19.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BibTeX `apalike' bibliography style (24-Jan-88 version)
% Adapted from the `alpha' style, version 0.99a; for BibTeX version 0.99a.
% Copyright (C) 1988, all rights reserved.
% Copying of this file is allowed, provided that if you make any changes at all
% you name it something other than `apalike.bst'.
% This restriction helps ensure that all copies are identical.
% Differences between this style and `alpha' are generally heralded by a `%'.
% The file btxbst.doc has the documentation for alpha.bst.
%
% This style should be used with the `apalike' LaTeX style (apalike.sty).
% \cite's come out like "(Jones, 1986)" in the text but there are no labels
% in the bibliography, and something like "(1986)" comes out immediately
% after the author.  Author (and editor) names appear as last name, comma,
% initials.  A `year' field is required for every entry, and so is either
% an author (or in some cases, an editor) field or a key field.
%
% Editorial note:
% Many journals require a style like `apalike', but I strongly, strongly,
% strongly recommend that you not use it if you have a choice---use something
% like `plain' instead.  Mary-Claire van Leunen (A Handbook for Scholars,
% Knopf, 1979) argues convincingly that a style like `plain' encourages better
% writing than one like `apalike'.  Furthermore the strongest arguments for
% using an author-date style like `apalike'---that it's "the most practical"
% (The Chicago Manual of Style, University of Chicago Press, thirteenth
% edition, 1982, pages 400--401)---fall flat on their face with the new
% computer-typesetting technology.  For instance page 401 anachronistically
% states "The chief disadvantage of [a style like `plain'] is that additions
% or deletions cannot be made after the manuscript is typed without changing
% numbers in both text references and list."  LaTeX sidesteps the disadvantage.
%
% History:
%   15-sep-86	(SK,OP)	Original version, by Susan King and Oren Patashnik.
%   10-nov-86	(OP)	Truncated the sort.key$ string to the correct length
%			in bib.sort.order to eliminate error message.
%   24-jan-88	(OP)	Updated for BibTeX version 0.99a, from alpha.bst 0.99a;
%			apalike now sorts by author, then year, then title;
%			THIS `apalike' VERSION DOES NOT WORK WITH BIBTEX 0.98i.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Allan A. Sioson (virginia tech, va 24061-0106, asioson@vt.edu)
% added the following modifications 
%
%   2004 September 21 (AAS) Bioinformatics Journal uses 'and' instead of &
%                           hence all & are reverted to 'and'.
%                     (AAS) Author names in Bioinformatics journal looks
%                           like 'Last,F.S.' and not like 'Last, F. S.'
%                           (i.e. spaces are removed). 
%
% The program was further modified by Tom Schneider:
%   1989 June 19 (TDS)  Renamed jmb.bst.  Items that do not have years were
%        listed in the bibliography without identifiers.  However, they
%        have identifiers in the text.  The identifier was added.
%        Titles were removed :-(.
%
%   1989 October 2 (TDS)  Made labels in the paper use the & symbol rather
%                  than `and', as required by JMB.
%   1989 October 3 (TDS)  Made et~al. be in italics.
%                  Removed comma after journal name.  (actually, all
%                  mid sentence commas go!)
%                  Made volume number be bold faced.
%                  Converted colon (:) after volume number to a comma (,).
%                  If a journal name does NOT end in a period, add
%                  a comma after the name.
%   1989 October 14 (TDS)  Without a year the program used to sort on the
%                   title.  I made it sort on the cite$ if there is no year.
%   1989 October 18 (TDS)  If there is one reference without a year, put
%                   an "a" at the end of its label, so that it comes out
%                   as (Smith, a) in the text.  in FUNCTION {forward.pass}
%   1989 November 2 (TDS)  Unpublished papers now give the title.
%   1989 November 18 (TDS)  TITLES can be turned on and off!
%                   Just nocite the reference TitlesOn in the paper
%                   and have a bibliography article in your database
%                   with that cite key!
%   1989 December 5 (TDS)  I can't stand sorting to fall back on the
%                   titles of the article - no control!  So in function
%                   bib.sort.order the default is now cite$.
%   1990 November 14 (TDS) Changes to match JTB better:
%                    Names in the bibliography are now connected
%                    with "\&" instead of "and", since this fits JTB better.
%                    "In" is now "In:".
%                    "editors" and "editor" are now "eds" and "ed".
%                    booktitle follows editors.  editors in parenthesis.
%                    pp. and p. instead of pages and page.
%                    location followed by colon (:) and then publisher.
%   1992 October 22  (TDS) Publisher address is at the end of books.
%                    with no colon after address.
%                    no comma before & in name list in bibliography.
%                    The extra.label (as the a of 1991a) is emphasized.
%                    "In:" goes back to "In"!
%                    publisher address is at the end of inproceedings.
%                    volume is "vol.", with comma after number.
%                    ed.", and comma after ) closing editors.
%                    as a colon (:), JMB does NOT want the next
%                    ed, but the format.title function uses
%                    which leaves it capitalized.  To fix this, a
%                    place change.case$ must be written.
%
%                    When a title has a colon (:), JMB does NOT want the next
%                    word capitalized, but the format.title function uses
%                    change.case$, which leaves it capitalized.  To fix this, a
%                    function to replace change.case$ was written:
%                    special.change.case
%
%                    edition should follow TITLE for books!

% possible bug:  sorting order probably should be first author followed
% by year; but this is not absolutely clear.  Sorting here is (I think)
% by all authors (or perhaps the key) then year.


ENTRY  % declare variables that have a value for each entry on the list
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
%    month                not used in apalike
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
  }
  {}
  { label extra.label sort.label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block
           docomma givetitles }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=

  #0 'docomma :=    % if it is 0 then don't do commas, otherwise do them.
}

FUNCTION {init.toggle.switches}
{% set switches for controlling the output!
  #1 'givetitles := % if it is 0 then don't give titles, otherwise do them.
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =

% %    { ", " * write$ }  % the comma here is responsible for every comma!
% %                       % But JMB doesn't want commas, so away it goes!
%      { " " * write$ }    % That does it!

    { % doing a comma is controlled specifically in JMB using docomma
      docomma #0 =
      { " " * write$ }
      { ", " * write$ }
      if$
    }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
     {  t "title" =
           { pop$ } % jmb ignores titles and does not object if missing
           { pop$ "empty " t * " in " * cite$ * warning$ }
        if$
     }
    'output.nonnull  % block periods
    % { pop$  } % don't do anything
  if$
}

%    t "title" =
%    { "zowie" warning$ }
%    {  duplicate$ empty$
%       { pop$ "EmPtY " t * " in " * cite$ * warning$ }
%       'output.nonnull
%    }
%    if$
%  if$
%}

%                			apalike needs this function because
%					the year has special punctuation;
%					apalike ignores the month
FUNCTION {output.year.check}
{ year empty$
    {
      "empty year in " cite$ ", using label: " extra.label * * * warning$
      write$
%      " (" extra.label * ")" * % TDS UNemphasized label

      " ({\em{" extra.label * "}})" * % TDS emphasized label

    { label "{\em{" extra.label "}}" * * * 'label :=}

      mid.sentence 'output.state :=
    }
    { write$

%      " (" year * extra.label * ")" * % TDS UNemphasized label
      " (" year * "{\em{" extra.label "}}" * * * ")" * % TDS emphasized label

      mid.sentence 'output.state :=
    }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$
  "]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
	'skip$
	{ after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
%    { s nameptr "{vv~}{ll}{, jj}{, f.}" format.name$ 't :=   % last name first
    { s nameptr "{vv~}{ll}{,jj}{,f{.}.}" format.name$ 't :=   % last name first
    % in bioinformatics, Author names look like 'Last,F.S.' 
    % i.e. no space in between the last name and first name initial
    % and between the first name and second name initials. -allan-
      nameptr #1 >
	{ namesleft #1 >
	    { ", " * t * }
	    { numnames #2 >
%		{ "," * } % remove this comma TDS
		{ " " * }
		'skip$
	      if$
	      t "others" =
		{ " {\em et~al.}" * }
%		{ " \& " * t * }
		{ " and " * t * }
	      if$
	    }
	  if$
	}
	't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.key}			% this function is just for apalike
{ empty$
    { key field.or.null }
    { "" }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
	{ ", eds" * } % TDS
	{ ", ed." * } % TDS
      if$
    }
  if$
}

FUNCTION {special.change.case}
% change the entire string to lower case except
% for the first letter and things inside {}
% That is, force the first letter after a colon (:) to be lower case
% but otherwise use the change.case$ rules.
{
% title "t" change.case$ % TDS this line was the original method

% start with the first letter of the title, hang on stack
title #1 #1 substring$

% unearth this to see the breakpoint
%"|---|" *

% take the title, change to entire lower case.
title "l" change.case$

% take from second position of lowered title to end of title
#2
global.max$
substring$

% concatenate first letter to rest of title
*
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { % decide whether to give the title or not
       givetitles #1 =
%       { title "t" change.case$ } % produce the title % TDS original function
       { special.change.case } % produce the title
       { title pop$ "" }          % don't produce the title
       if$
    }
  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
	{ t #1 #2 substring$ "--" = not
	    { "--" *
	      t #2 global.max$ substring$ 't :=
	    }
	    {   { t #1 #1 substring$ "-" = }
		{ "-" *
		  t #2 global.max$ substring$ 't :=
		}
	      while$
	    }
	  if$
	}
	{ t #1 #1 substring$ *
	  t #2 global.max$ substring$ 't :=
	}
      if$
    }
  while$
}

FUNCTION {format.btitle}
{ title emphasize
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
%    { "volume" volume tie.or.space.connect % TDS
    { "vol." volume "," * tie.or.space.connect % made vol.
      series empty$
	'skip$
	{ " of " * series emphasize * }
      if$
%      "volume and number" number either.or.check % TDS
      "vol. and num." number either.or.check % made vol.
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
	{ series field.or.null }
	{ output.state mid.sentence =
	    { "number" }
	    { "Number" }
	  if$
	  number tie.or.space.connect
	  series empty$
	    { "there's a number but no series in " cite$ * warning$ }
	    { " in " * series * }
	  if$
	}
      if$
    }
    { "" }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { output.state mid.sentence =
	{ edition "l" change.case$ " edition," * }         % added for CABIOS
	{ edition "t" change.case$ " edition," * }         % added for CABIOS
      if$
    }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
	{ #1 'multiresult := }
	{ t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
	{ "pp." pages n.dashify tie.or.space.connect } % TDS
	{ "p." pages tie.or.space.connect } % TDS
      if$
    }
  if$
}

FUNCTION {format.vol.num.pages}
{ "{\bf " volume "}" * * field.or.null  % make volume bold face
  number empty$
    'skip$
    { " (" number * ")" * *
       volume empty$
	{ "there's a number but no volume in " cite$ * warning$ }
	'skip$
      if$
    }
  if$
  pages empty$
    'skip$
    { duplicate$ empty$
	{ pop$ format.pages }
%	{ ":" * pages n.dashify * } % no more colons after the volume
	{ ", " * pages n.dashify * } % comma's instead
      if$
    }
  if$
}


FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
	{ "chapter" }
	{ type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
	'skip$
	{ ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
    { "" }
    { editor empty$
	{ "In " booktitle emphasize * } % TDS
	{ "In " booktitle emphasize * ", " * % That's the comma after the title
          "(" format.editors ")," * * * } % TDS
      if$
    }
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "Technical Report" }
    'type
  if$
  number empty$
    { "t" change.case$ }
    { number tie.or.space.connect }
  if$
}

FUNCTION {format.article.crossref}
{ "In"				% this is for apalike % TDS
  " \cite{" * crossref * "}" *
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "In " % TDS
    }
    { "Volume" volume tie.or.space.connect
      " of " *
    }
  if$
  "\cite{" * crossref * "}" *				% this is for apalike
}

FUNCTION {format.incoll.inproc.crossref}
{ "In"				% this is for apalike % TDS
  " \cite{" * crossref * "}" *
}

FUNCTION {article}
{
  cite$ "TitlesOn" =
  'skip$ % Don't write out an article of this kind, it's a toggle switch!
  {
  output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
%  new.block                                            %change for CABIOS don't use period after year
  format.title "title"  output.check
  new.block
  crossref missing$
    { % figure out the last character of the journal name.
      % "<<" journal #-1 #1 substring$ ">>" * *  % for testing
      % "<" journal ">" * *  % this works ok     % for testing

      journal #-1 #1 substring$ "." =
      { journal ", " *}  % journal ended in period so add comma too
      { journal ", " *}  % journal ended without period, so add comma
      if$
      emphasize "journal" output.check

%      journal * emphasize "journal" output.check % the original method

      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
  }
  if$
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  output.year.check				% special for apalike
%  new.block                                    %change for CABIOS don't use period after year
  format.btitle "title" output.check
  #1 'docomma :=    % TURN COMMAS ON FOR JMB
  crossref missing$
    {
      format.bvolume output
      new.block
      format.number.series output
      format.edition output % TDS JMB position for edition
%      new.sentence
%         #0 'docomma := % no comma after address
%      address ":" * output % TDS
      publisher "publisher" output.check % TDS
%         #1 'docomma := % comma back on
%      publisher "publisher" output.check
%      address output
      address output % TDS - no colon!
    }
    { new.block
      format.book.crossref output.nonnull
    }
  if$
%  format.edition output % TDS original position of edition
  new.block
  note output
  fin.entry
  #0 'docomma :=    % TURN COMMAS OFF
}

FUNCTION {booklet}
{ output.bibitem
  format.authors output
  author format.key output				% special for
  output.year.check					% apalike
  new.block
  format.title "title" output.check
  new.block
  howpublished output
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  output.year.check				% special for apalike
%  new.block                                    % changed for CABIOS no period after year
  format.btitle "title" output.check
  crossref missing$
    { format.bvolume output
%      format.chapter.pages "chapter and pages" output.check
%      new.block
      format.number.series output
      new.sentence
         #0 'docomma := % no comma after address
      address ":" * output % TDS
      publisher "publisher" output.check % TDS
         #0 'docomma := % comma back on            %changed to keep comma's off for CABIOS
%      publisher "publisher" output.check
%      address output
    }
    { format.chapter.pages "chapter and pages" output.check
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
   
%  new.block                                            %removed period for CABIOS
  format.pages output                                   %added for CABIOS
  note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
%  new.block                                            % changed for CABIOS no period after year
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
%      format.chapter.pages output                      %changed for CABIOS 
      new.sentence
      publisher "publisher" output.check
      address output
      format.edition output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  format.pages output                                   %added for CABIOS
%  new.block                                            %removed for CABIOS
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
%  new.block                                            % no period after year in CABIOS
  format.title "title" output.check
  new.block
  crossref missing$
    {
%      #1 'docomma :=    % TURN COMMAS ON FOR JMB, BUT NOT FOR JTB: inconsistent. TDS NOW
      format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      format.pages
        pages empty$
          { output }
%          { "," * output } % TDS toss in comma instead of period
          { output }
        if$

%      address output % TDS address is below for JTB
%      new.sentence % TDS remove period
      organization output
         #0 'docomma := % NO COMMA AFTER ADDRESS
      publisher "publisher" output.check % TDS
         #1 'docomma := % comma back on                 

%      publisher output					% are simpler
      address output % TDS
      #0 'docomma :=    % TURN COMMAS OFF
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  format.authors output
  author format.key output				% special for
  output.year.check					% apalike
%  new.block                                            % change for CABIOS don't use period after year
  format.btitle "title" output.check
  organization address new.block.checkb
  organization output
  address output
  format.edition output
  new.block
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
  new.block
  format.title "title" output.check
  new.block
  "Master's thesis" format.thesis.type output.nonnull
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  author format.key output				% special for
  output.year.check					% apalike
  new.block
  format.title output
  new.block
  howpublished output
  new.block
  note output
  fin.entry
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
  new.block
  format.btitle "title" output.check
  new.block
  "PhD thesis," format.thesis.type output.nonnull
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  format.editors output
  editor format.key output				% special for
  output.year.check					% apalike
  new.block
  format.btitle "title" output.check
  format.bvolume output
  format.number.series output
  address output				% for apalike
  new.sentence					% we always output
  organization output				% a nonempty organization
  publisher output				% here
  new.block
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
  output.year.check					% apalike
  new.block
  format.title "title" output.check
  new.block
  format.tr.number output.nonnull
  institution "institution" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  author format.key output				% special for
%  output.year.check					% apalike
  new.block
% zzz
  format.title "title" output.check

% Since format.title is out of commission, the original method won't work:
%  format.title "title" output.check
% so do the equivalent of the format.title procedure: (TDS)

%  title empty$
%    { "" }
%    { title "t" change.case$ } % produce the title
%  if$
%  "title" output.check

  new.block
  note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"January"}

MACRO {feb} {"February"}

MACRO {mar} {"March"}

MACRO {apr} {"April"}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {oct} {"October"}

MACRO {nov} {"November"}

MACRO {dec} {"December"}

MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}

READ

% ***********************************************************

FUNCTION {check.for.titlecommand}
% go through each entry and see if one of them has the key
% 'TitlesOn'.  If this is found, then do titles!
% 1989 November 18 TDS
{
%  "|" label "|" * * write$ newline$  % |Arrhenius {\em et~al.}, 1986|
%  "|" cite$ "|" * * write$ newline$  % |Arrhenius1986|
% so cite$ is the thing I want to detect...

  cite$ "TitlesOn" =
  { % got it!
%    "FOUND TitlesOn" write$ newline$
     "Titles Will Be Printed" warning$
    #1 'givetitles := % give titles!
  }
  'skip$
  if$
}
EXECUTE {init.toggle.switches}
ITERATE {check.for.titlecommand}
% ***********************************************************

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

%			There are three apalike cases: one person (Jones),
%			two (Jones and de~Bruijn), and more (Jones et~al.).
%			This function is much like format.crossref.editors.
%
FUNCTION {format.lab.names}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " {\em et~al.}" * }
    { #2 <
	'skip$
	{ s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
	    { " {\em et~al.}" * }
%	    { " \& " * s #2 "{vv~}{ll}" format.name$ * }
            % note new use of & rather than `and'.  TDS
	    { " and " * s #2 "{vv~}{ll}" format.name$ * }
	    % I brought & back to 'and'.  AAS
	  if$
	}
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
	{ cite$ #1 #3 substring$ }
	'key					% apalike uses the whole key
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
	{ key empty$
	    { cite$ #1 #3 substring$ }
	    'key				% apalike uses the whole key
	  if$
	}
	{ editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.label}
{ editor empty$
    { key empty$
	{ cite$ #1 #3 substring$ }
	'key			% apalike uses the whole key, no organization
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.label}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
	'editor.key.label			% apalike ignores organization
	'author.key.label			% for labeling and sorting
      if$
    }
  if$
  ", "							% these three lines are
  *							% for apalike, which
  year field.or.null purify$ #-1 #4 substring$		% uses all four digits
  *
  'label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
	{ "   " * }
	'skip$
      if$						% apalike uses initials
      s nameptr "{vv{ } }{ll{ }}{  f{ }}{  jj{ }}" format.name$ 't := % <= here
      nameptr numnames = t "others" = and
	{ "{\em et al}" * }
	{ t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
	{ "to sort, need author or key in " cite$ * warning$
	  ""
	}
	{ key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
	{ key empty$
	    { "to sort, need author, editor, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.sort}
{ editor empty$
    { key empty$
	{ "to sort, need editor or key in " cite$ * warning$
	  ""
	}
	{ key sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

%			apalike uses two sorting passes; the first one sets the
%			labels so that the `a's, `b's, etc. can be computed;
%			the second pass puts the references in "correct" order.
%			The presort function is for the first pass. It computes
%			label, sort.label, and title, and then concatenates.
FUNCTION {presort}
{ calc.label
  label sortify
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
	'editor.sort
	'author.sort
      if$
    }
  if$
  #1 entry.max$ substring$	% for
  'sort.label :=		% apalike
  sort.label			% style
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT		% by label, sort.label, title---for final label calculation

STRINGS { last.label next.extra }	% apalike labels are only for the text;

INTEGERS { last.extra.num }		% there are none in the bibliography

FUNCTION {initialize.extra.label.stuff}	% and hence there is no `longest.label'
{ #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'last.extra.num :=
}

FUNCTION {forward.pass}
% pass through the references forward
{
%"\\     label=" label "//" * * write$ newline$ % display the label
%"\\last.label=" last.label "//" * * write$ newline$ % display the label

 last.label label =   % if the label repeats the previous label...
    { % then increment the extra number
      last.extra.num #1 + 'last.extra.num :=
      % and convert it to a label
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { % else set things up for the next entry
      "a" chr.to.int$ 'last.extra.num :=

      % However, if the year was missing, tack on an extra "a". TDS
      year empty$
      {"a" 'extra.label :=}
      {"" 'extra.label :=}
      if$

%      "" 'extra.label :=  % the original method was not to do anything TDS

      label 'last.label :=   % capture this label for next time
    }
  if$
%"\\extra.label = " extra.label "//" * * write$ newline$ % display the label
}

FUNCTION {reverse.pass}
% pass through the references backwards
% add extra characters to the end of the label string
{
%"{{" label "}}" * * write$ newline$ % display the label
 next.extra "b" =
% original logic:
    { "a" 'extra.label := }
    'skip$
    if$

% next.extra "b" = { "next.extra was = b" write$ newline$} 'skip$ if$

% new as of 1989 Oct 18
%    { "a" 'extra.label := }
%    { % Put on an "a" at the end of the label if the year is missing. TDS
%      year empty$
%       { "a" 'extra.label := }
%       %{ label "a" * 'label := }
%       'skip$ % otherwise leave it alone (orignial method)
%     if$ }
%  if$

%  label extra.label * 'label := % TDS ORIGINAL LABEL METHOD

% stick {\em } around extra.label if it's not empty TDS
  extra.label "" =
    'skip$
    { label "{\em{" extra.label "}}" * * * 'label :=}
  if$

  extra.label 'next.extra :=
%"{{" label "}}" * * write$ newline$ % display the label
}

FUNCTION {bib.sort.order}
% Generate the sort.key$ variables for sorting.
% The sorting is first on the sort.label (ie, author's name), followed
% by the year then the title.  If there is no year, the cite$ is used.
{
  sort.label  % this is based on the author name
  "    " * % tack on some space
% original apa command was to use the year or an empty string:
%  year field.or.null sortify
% 
% Replace that with the use of the cite$, when there is no year:
  year duplicate$ empty$
     {pop$ cite$}
     'skip$
  if$

  * "    " *  % tack on some space

% I CAN'T STAND SORTING ON TITLE!!! TDS 1989 Dec 5
%  title field.or.null
%  sort.format.title
%  * % attach them together
  cite$
  * % use cite$ instead!!

  % note: if there is no year, then the cite$ will override the sorting
  % on the title.  Oh well.  Title sorting is sorta (ha ha) awful.

  #1 entry.max$ substring$
  'sort.key$ :=

  % The following line helps debug the program.  It shows what the sort.key$ is.
  % "%" sort.key$ * write$ newline$
}

% Here is the place that the actual executions of the labeling and sorting
% functions are done.


EXECUTE {initialize.extra.label.stuff} % initialize variables
ITERATE {bib.sort.order}               % set up the sorting keys
SORT		% by sort.label, year, title---giving final bibliography order
% Having sorted NOW we apply the extra letters at the end!
ITERATE {forward.pass}
REVERSE {reverse.pass}

FUNCTION {begin.bib}
{ preamble$ empty$				% no \etalchar in apalike
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{}" write$ newline$		% no labels in apalike
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
