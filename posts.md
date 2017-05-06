---
layout: page
title: Posts
---

<!--
{% include top_tags.html count_as_heading='Five' count_as_number=5 %}
{% include post_archive.html %}
-->
<div>
  <ul class="posts">
    {% for post in site.posts %}
      {% assign post_date = post.date | date_to_string %}
      {% capture post_url %} {{ site.baseurl }}{{ post.url }} {% endcapture %}
      <li>
	    <span class="entry-date">{{ post_date }}<br>
	    <a href="{{ post_url }}">{{ post.title }}</a></span>
	 </li>
    {% endfor %}
  </ul>
</div>

