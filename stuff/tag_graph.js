/**
 * Created by guydavidson on 23/02/2016.
 */

var tagColorMap = {};
var tagColorScale = d3.scale.category20();

function tagColor(tag) {
    var tagId = tag.id;

    if (tagId in tagColorMap) {
        return tagColorMap[tagId];
    }

    var color = tagColorScale(Math.floor((Math.random() * 20) + 1));
    tagColorMap[tagId] = color;
    return color;
}

var linkColorMap = {};
var linkColorScale = d3.scale.category20b();

function linkColor(link) {
    var linkKey = link.source.id + link.target.id;

    if (linkKey in linkColorMap) {
        return linkColorMap[linkKey];
    }

    var color = linkColorScale(Math.floor((Math.random() * 20) + 1));
    linkColorMap[linkKey] = color;
    return color;
}


function tagGraph(graphFile, linkLength, minNode, width, height) {
    if (width == undefined) {
        width = $(document).width();
    }

    if (height == undefined) {
        height = $(document).height();
    }

    var force = cola.d3adaptor()
        // .linkDistance(function (d) {
        //   return Math.sqrt(d.submissions.length) * 40;
        // })
        .symmetricDiffLinkLengths(linkLength)
        .avoidOverlaps(true)
        .size([width, height]);

    // var force = d3.layout.force()
    //     .charge(function (n) {
    //       return (Math.log(n.submissions.length) * 10) - 100;
    //     })
    //     .linkDistance(400)
    //     .linkStrength(function (d) {
    //       return d.submissions.length > 20 ? .5 : .1;
    //     })
    //     .gravity(0.05)
    //     .size([width, height]);

    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height);

    var tip = d3.tip().attr('class', 'd3-tip').html(function(d) { return d.name; });

    svg.call(tip);

    d3.json(graphFile, function(error, graph) {
        if (error) throw error;

        var vertices = graph.nodes.filter(function(n) {
            return n.submissions.length >= minNode;
        });

        var edges = [];
        graph.links.forEach(function(e) {
            var sourceNode = vertices.filter(function(n) {return n.id === e.source; })[0];
            var targetNode = vertices.filter(function(n) {return n.id === e.target; })[0];

            if (sourceNode && targetNode && e.submissions.length > 0) {
                edges.push({
                    source: sourceNode,
                    target: targetNode,
                    submissions: e.submissions,
                    name: e.submissions.map(function (s) { return graph.submissions[s].title; }).join("<br>"),
                });
            }
        });

        vertices = vertices.filter(function (n) {
          return edges.filter(function (e) {
            return e.source == n || e.target == n
          }).length != 0
        });

        force
          .nodes(vertices)
          .links(edges)
          .start(100, 100, 100);

         var link = svg.selectAll(".link")
                 .data(edges)
                 .enter().append("line")
                 .attr("class", "link")
                 .style("stroke", linkColor)
                 .style("stroke-width", function(d) { return d.submissions.length * 1.5; })
                 .on("mouseover", tip.show)
                 .on("mouseout", tip.hide);

        link.append("title").text(function(d) { return d.submissions;});

        var node = svg.selectAll(".node")
                .data(vertices)
                .enter()
                .append("circle")
                .attr("class", "node")
                .attr("r", function(d) { return Math.log(d.submissions.length) * 5; })
                .style("fill", tagColor)
                .on("mouseover", tip.show)
                .on("mouseout", tip.hide)
                .call(force.drag);

        var text = svg.selectAll(".node")
                .data(vertices)
                .enter()
                .append("text")
                .attr("dy", ".3em")
                .style("text-anchor", "middle")
                .text(function(d) { return d.name; });

        force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                  .attr("y1", function(d) { return d.source.y; })
                  .attr("x2", function(d) { return d.target.x; })
                  .attr("y2", function(d) { return d.target.y; });

            node.attr("cx", function(d) { return d.x; })
                    .attr("cy", function(d) { return d.y; });
        });
    });
}
